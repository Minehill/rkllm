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

#define PROMPT_TEXT_PREFIX "<用户>"
#define PROMPT_TEXT_POSTFIX "<AI>"


using namespace std;
LLMHandle llmHandle = nullptr;

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
        // for(int i=0; i<result->num; i++)
        // {
        //     printf("%d token_id: %d logprob: %f\n", i, result->tokens[i].id, result->tokens[i].logprob);
        // }
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

    //设置参数及初始化
    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = argv[1];
    param.top_k = 1;
    param.max_new_tokens = std::atoi(argv[2]);
    param.max_context_len = std::atoi(argv[3]);

    // if use multimodel mode, need to set img_start,img_end and img_content
    param.img_start = "<image>";
    param.img_end = "</image>\n";
    param.img_content = "<unk>";

    param.skip_special_token = true;

    int ret = rkllm_init(&llmHandle, &param, callback);
    if (ret == 0){
        printf("rkllm init success\n");
    } else {
        printf("rkllm init failed\n");
        exit_handler(-1);
    }

    vector<string> pre_input;
    pre_input.push_back("<image>What is in the image?");
    cout << "\n**********************可输入以下问题对应序号获取回答/或自定义输入********************\n"
         << endl;
    for (int i = 0; i < (int)pre_input.size(); i++)
    {
        cout << "[" << i << "] " << pre_input[i] << endl;
    }
    cout << "\n*************************************************************************\n"
         << endl;

    string text;
    RKLLMInput rkllm_input;

    // 初始化 infer 参数结构体
    RKLLMInferParam rkllm_infer_params;
    memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));  // 将所有内容初始化为 0

    // // 1. 初始化并设置 LoRA 参数（如果需要使用 LoRA）
    // RKLLMLoraParam lora_params;
    // lora_params.lora_adapter_name = "test";  // 指定用于推理的 lora 名称

    // // 2. 初始化并设置 Prompt Cache 参数（如果需要使用 prompt cache）
    // RKLLMPromptCacheParam prompt_cache_params;
    // prompt_cache_params.save_prompt_cache = true;                  // 是否保存 prompt cache
    // prompt_cache_params.prompt_cache_path = "./prompt_cache.bin";  // 若需要保存prompt cache, 指定 cache 文件路径

    rkllm_infer_params.mode = RKLLM_INFER_GENERATE;
    // rkllm_infer_params.lora_params = &lora_params;
    // rkllm_infer_params.prompt_cache_params = &prompt_cache_params;

    // rkllm_load_prompt_cache(llmHandle, "./prompt_cache.bin");
    while (true)
    {
        std::string input_str;
        printf("\n");
        printf("user: ");
        std::getline(std::cin, input_str);
        if (input_str == "exit")
        {
            break;
        }
        for (int i = 0; i < (int)pre_input.size(); i++)
        {
            if (input_str == to_string(i))
            {
                input_str = pre_input[i];
                cout << input_str << endl;
            }
        }

        float * img_embed_data_ptr = (float *)malloc(64 * 2304 * sizeof(float));
        // 打开输入文件流（以二进制模式）
        std::ifstream inFile("./img_vec.bin", std::ios::binary);
        // 检查文件是否成功打开
        if (!inFile) {
            printf("Failed to open file for reading: ");
        }
        // 读取数据
        float temp_data;
        int idx = 0;
        while (inFile.read(reinterpret_cast<char*>(&temp_data), sizeof(float))){
            img_embed_data_ptr[idx] = temp_data;
            idx = idx + 1;
        }
        // 关闭文件
        inFile.close();

        text = PROMPT_TEXT_PREFIX + input_str + PROMPT_TEXT_POSTFIX;
        rkllm_input.input_type = RKLLM_INPUT_MULTIMODAL;
        rkllm_input.multimodal_input.prompt = (char *)text.c_str();
        rkllm_input.multimodal_input.image_embed = img_embed_data_ptr;
        rkllm_input.multimodal_input.n_image_tokens = 64;
        printf("robot: ");
        rkllm_run(llmHandle, &rkllm_input, &rkllm_infer_params, NULL);
        free(img_embed_data_ptr);
    }
    rkllm_destroy(llmHandle);

    return 0;
}