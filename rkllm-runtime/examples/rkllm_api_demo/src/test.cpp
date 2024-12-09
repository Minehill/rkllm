#include <iostream>
#include <string>
#include <vector>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#include <arrow/result.h>
#include <parquet/file_reader.h>

#include <fstream>
#include "rkllm.h"

int main(int argc, char **argv)
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " model_path max_new_tokens max_context_len\n";
        return 1;
    }

    // 初始化模型
    LLMHandle llmHandle = nullptr;
    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = argv[1];
    param.max_new_tokens = std::atoi(argv[2]);
    param.max_context_len = std::atoi(argv[3]);
    param.skip_special_token = true;

    int ret = rkllm_init(&llmHandle, &param, nullptr);
    if (ret != 0 || llmHandle == nullptr){
        std::cerr << "模型初始化失败" << std::endl;
        return -1;
    }

    // 读取测试集数据
    std::shared_ptr<arrow::io::ReadableFile> infile;
    auto result = arrow::io::ReadableFile::Open("test-00000-of-00001.parquet");
    if (!result.ok()) {
        std::cerr << "无法打开测试集文件。" << std::endl;
        return -1;
    }
    infile = *result;

    std::unique_ptr<parquet::arrow::FileReader> parquet_reader;
    arrow::Status status = parquet::arrow::FileReader::Open(infile, arrow::default_memory_pool(), &parquet_reader);
    if (!status.ok()) {
        std::cerr << "无法读取 Parquet 文件。" << std::endl;
        return -1;
    }

    std::shared_ptr<arrow::Table> table;
    status = parquet_reader->ReadTable(&table);
    if (!status.ok()) {
        std::cerr << "无法读取 Parquet 表格。" << std::endl;
        return -1;
    }

    // 获取表格列
    auto id_array = std::static_pointer_cast<arrow::StringArray>(table->GetColumnByName("id")->chunk(0));
    auto question_array = std::static_pointer_cast<arrow::StringArray>(table->GetColumnByName("question")->chunk(0));
    auto choices_array = std::static_pointer_cast<arrow::ListArray>(table->GetColumnByName("choices")->chunk(0));
    auto answerKey_array = std::static_pointer_cast<arrow::StringArray>(table->GetColumnByName("answerKey")->chunk(0));

    int64_t num_rows = table->num_rows();
    int correct_predictions = 0;

    // 遍历测试集
    for (int64_t i = 0; i < num_rows; ++i) {
        // 提取问题和正确答案
        std::string question = question_array->GetString(i);
        std::string answerKey = answerKey_array->GetString(i);

        // 提取选项
        std::vector<std::string> labels;
        std::vector<std::string> texts;

        auto choices = std::static_pointer_cast<arrow::StructArray>(choices_array->value_slice(i));
        auto label_array = std::static_pointer_cast<arrow::StringArray>(choices->GetFieldByName("label"));
        auto text_array = std::static_pointer_cast<arrow::StringArray>(choices->GetFieldByName("text"));

        for (int64_t j = 0; j < choices->length(); ++j) {
            labels.push_back(label_array->GetString(j));
            texts.push_back(text_array->GetString(j));
        }

        // 构建模型输入
        std::string model_input = question + "\n";
        for (size_t k = 0; k < labels.size(); ++k) {
            model_input += labels[k] + ". " + texts[k] + "\n";
        }
        model_input += "答案是：";

        // 调用模型进行推理
        std::string result_text;
        char result_buffer[1024] = {0};
        RKLLMInput rkllm_input;
        rkllm_input.input_type = RKLLM_INPUT_PROMPT;
        rkllm_input.prompt_input = (char *)model_input.c_str();

        RKLLMInferParam rkllm_infer_params;
        memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));
        rkllm_infer_params.mode = RKLLM_INFER_GENERATE;
        // rkllm_infer_params.output = result_buffer;
        // rkllm_infer_params.output_token_max = sizeof(result_buffer);


        ret = rkllm_run(llmHandle, &rkllm_input, &rkllm_infer_params, nullptr);
        if (ret != 0) {
            std::cerr << "模型推理失败。" << std::endl;
            continue;
        }

        result_text = result_buffer;

        // 简单处理模型输出，提取预测的答案
        std::string prediction;
        for (const auto& label : labels) {
            if (result_text.find(label) != std::string::npos) {
                prediction = label;
                break;
            }
        }

        // 如果无法匹配到标签，跳过
        if (prediction.empty()) {
            continue;
        }

        // 比较预测结果与正确答案
        if (prediction == answerKey) {
            correct_predictions++;
        }
    }

    // 计算准确率
    double accuracy = static_cast<double>(correct_predictions) / num_rows;
    std::cout << "模型在测试集上的准确率为：" << accuracy * 100 << "%" << std::endl;

    // 释放资源
    rkllm_destroy(llmHandle);

    return 0;
}