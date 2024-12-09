// #include <iostream>
// #include <fstream>
// #include <string>
// #include <vector>
// #include <nlohmann/json.hpp> // 需要安装 nlohmann/json 库

// using json = nlohmann::json;


// int main() {
//     std::ifstream infile("/userdata/repos/datasets/rkllm_code/rknn-llm/rkllm-runtime/examples/rkllm_api_demo/src/train_rand_split.jsonl");
//     if (!infile.is_open()) {
//         std::cerr << "无法打开文件" << std::endl;
//         return 1;
//     }

//     std::vector<std::string> questions;
//     std::vector<std::string> question_concepts;
//     std::vector<std::vector<std::string>> choices;
//     std::vector<std::string> answers;

//     std::string line;
//     while (std::getline(infile, line)) {
//         json j = json::parse(line);
//         questions.push_back(j["question"]["stem"]);
//         question_concepts.push_back(j["question"]["question_concept"]);
        
//         std::vector<std::string> choice_texts;
//         for (const auto& choice : j["question"]["choices"]) {
//             choice_texts.push_back(choice["text"]);
//         }
//         choices.push_back(choice_texts);
        
//         answers.push_back(j["answerKey"]);
//     }

//     infile.close();

//     输出结果以验证
//     for (size_t i = 0; i < questions.size(); ++i) {
//         std::cout << "问题: " << questions[i] << std::endl;
//         std::cout << "问题类别: " << question_concepts[i] << std::endl;
//         std::cout << "选项: ";
//         for (const auto& choice : choices[i]) {
//             std::cout << choice << " ";
//         }
//         std::cout << std::endl;
//         std::cout << "答案: " << answers[i] << std::endl;
//         std::cout << "------------------------" << std::endl;
//     }



//     return 0;
// }






// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string.h>
#include <unistd.h>
#include <string>
#include "rkllm.h"
#include <fstream>
#include <iostream>
#include <csignal>
#include <vector>
#include <nlohmann/json.hpp>

#define PROMPT_TEXT_PREFIX "<|im_start|>only answer letter number <|im_end|> <|im_start|>user"
#define PROMPT_TEXT_POSTFIX "<|im_end|><|im_start|>"


using json = nlohmann::json;
using namespace std;
LLMHandle llmHandle = nullptr;


std::vector<std::string> questions;
std::vector<std::string> question_concepts;
std::vector<std::vector<std::string>> choices;
std::vector<std::string> answers;
int correct_predictions = 0;
int total_predictions = 0;

// 读取json文件
void read_data () {
    std::ifstream infile("/userdata/repos/datasets/rkllm_code/rknn-llm/rkllm-runtime/examples/rkllm_api_demo/src/train_rand_split.jsonl");
    if (!infile.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
        return;
    }

    // std::vector<std::string> questions;
    // std::vector<std::string> question_concepts;
    // std::vector<std::vector<std::string>> choices;
    // std::vector<std::string> answers;

    std::string line;
    while (std::getline(infile, line)) {
        json j = json::parse(line);
        questions.push_back(j["question"]["stem"]);
        question_concepts.push_back(j["question"]["question_concept"]);
        
        std::vector<std::string> choice_texts;
        for (const auto& choice : j["question"]["choices"]) {
            // choice_texts.push_back(choice["text"]);
            choice_texts.push_back(choice["label"].get<std::string>() + " : " + choice["text"].get<std::string>());
        }
        choices.push_back(choice_texts);
        
        answers.push_back(j["answerKey"]);
    }

    infile.close();

    // 输出结果以验证
    // for (size_t i = 0; i < questions.size(); ++i) {
    //     std::cout << "问题: " << questions[i] << std::endl;
    //     std::cout << "问题类别: " << question_concepts[i] << std::endl;
    //     std::cout << "选项: ";
    //     for (const auto& choice : choices[i]) {
    //         std::cout << choice << " ";
    //     }
    //     std::cout << std::endl;
    //     std::cout << "答案: " << answers[i] << std::endl;
    //     std::cout << "------------------------" << std::endl;
    // }
}


void exit_handler(int signal)
{
    if (llmHandle != nullptr)
    {
        {
            cout << "程序即将退出" << endl;
            LLMHandle _tmp = llmHandle;
            llmHandle = nullptr;
            rkllm_destroy(_tmp);
        }
    }
    exit(signal);
}

void callback(RKLLMResult *result, void *userdata, LLMCallState state)
{
    if (state == RKLLM_RUN_FINISH)
    {
        printf("\n");
    } else if (state == RKLLM_RUN_ERROR) {
        printf("\\run error\n");
    } else if (state == RKLLM_RUN_GET_LAST_HIDDEN_LAYER) {
        /* ================================================================================================================
        若使用GET_LAST_HIDDEN_LAYER功能,callback接口会回传内存指针:last_hidden_layer,token数量:num_tokens与隐藏层大小:embd_size
        通过这三个参数可以取得last_hidden_layer中的数据
        注:需要在当前callback中获取,若未及时获取,下一次callback会将该指针释放
        ===============================================================================================================*/
        if (result->last_hidden_layer.embd_size != 0 && result->last_hidden_layer.num_tokens != 0) {
            int data_size = result->last_hidden_layer.embd_size * result->last_hidden_layer.num_tokens * sizeof(float);
            printf("\ndata_size:%d",data_size);
            std::ofstream outFile("last_hidden_layer.bin", std::ios::binary);
            if (outFile.is_open()) {
                outFile.write(reinterpret_cast<const char*>(result->last_hidden_layer.hidden_states), data_size);
                outFile.close();
                std::cout << "Data saved to output.bin successfully!" << std::endl;
            } else {
                std::cerr << "Failed to open the file for writing!" << std::endl;
            }
        }
    } else if (state == RKLLM_RUN_NORMAL) {
        printf("%s", result->text);
        // 将第一个字母与答案进行比较（忽略大小写差异），计数正确与错误的情况
        if (result->text[1] == answers[total_predictions][0] || result->text[1] == answers[total_predictions][0] + 32) {
            correct_predictions++;
            printf("yes");
        }
        total_predictions++;
    }
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " model_path max_new_tokens max_context_len\n";
        return 1;
    }

    signal(SIGINT, exit_handler);
    printf("rkllm init start\n");
    
    // 读取json文件
    read_data();

    //设置参数及初始化
    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = argv[1];

    //设置采样参数
    param.top_k = 1;
    param.top_p = 0.95;
    param.temperature = 0.8;
    param.repeat_penalty = 1.1;
    param.frequency_penalty = 0.0;
    param.presence_penalty = 0.0;

    param.max_new_tokens = std::atoi(argv[2]);
    param.max_context_len = std::atoi(argv[3]);
    param.skip_special_token = true;
    param.extend_param.base_domain_id = 0;

    int ret = rkllm_init(&llmHandle, &param, callback);
    if (ret == 0){
        printf("rkllm init success\n");
    } else {
        printf("rkllm init failed\n");
        exit_handler(-1);
    }

    string text;
    RKLLMInput rkllm_input;

    // 初始化 infer 参数结构体
    RKLLMInferParam rkllm_infer_params;
    memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));  // 将所有内容初始化为 0

    rkllm_infer_params.mode = RKLLM_INFER_GENERATE;
    
    printf("size : %d", questions.size());

    for (size_t i = 0; i < questions.size(); ++i)
    {
        text = questions[i] + "\n";
        for (size_t k = 0; k < choices[i].size(); ++k) {
            text += choices[i][k] + "\n";
        }
        text += "答案是(仅字母):";
        
        // text = PROMPT_TEXT_PREFIX + text + PROMPT_TEXT_POSTFIX;
        // std::cout <<  text << std::endl;

        rkllm_input.input_type = RKLLM_INPUT_PROMPT;
        rkllm_input.prompt_input = (char *)text.c_str();
        // printf("robot: ");

        // 若要使用普通推理功能,则配置rkllm_infer_mode为RKLLM_INFER_GENERATE或不配置参数
        rkllm_run(llmHandle, &rkllm_input, &rkllm_infer_params, NULL);
    }
    rkllm_destroy(llmHandle);

    // 计算准确率
    double accuracy = static_cast<double>(correct_predictions) / total_predictions;
    std::cout << "模型在测试集上的准确率为：" << accuracy * 100 << "%" << std::endl;


    return 0;
}