//
// Created by hanke on 2024/4/21.
//
#include <cmath>
#include "maxpooling.hpp"
#include "node/abstract/node_factory.hpp"
#include "infer/infer_ir.hpp"
namespace infer_neto {
InferStatus MaxPoolingLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                     std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
    if (inputs.empty()) {
        LOG(ERROR) << "The input tensor array in the max pooling layer is empty";
        return InferStatus::kInferFailedInputEmpty;
    }

    if (inputs.size() != outputs.size()) {
        LOG(ERROR)
                << "The input and output tensor array size of the max pooling layer "
                   "do not match";
        return InferStatus::kInferFailedInputOutSizeMatchError;
    }

    const uint32_t batch = inputs.size();
    const uint32_t pooling_h = pooling_size_h_;
    const uint32_t pooling_w = pooling_size_w_;
    if (!stride_h_ || !stride_w_) {
        LOG(ERROR) << "The stride parameter is set incorrectly. It must always be "
                      "greater than 0";
        return InferStatus::kInferFailedStrideParameterError;
    }
    for (uint32_t i = 0; i < batch; ++i) {
        const std::shared_ptr<ftensor>& input_data = inputs.at(i);
        if (input_data == nullptr || input_data->empty()) {
            LOG(ERROR) << "The input tensor array in the max pooling layer has an "
                          "empty tensor "
                       << i << "th";
            return InferStatus::kInferFailedInputEmpty;
        }
        uint32_t input_h = input_data->rows();
        uint32_t input_w = input_data->cols();
        uint32_t output_h = uint32_t(std::floor(
                (int(input_h) - int(pooling_h) + 2 * padding_h_) / stride_h_ + 1));
        uint32_t output_w = uint32_t(std::floor(
                (int(input_w) - int(pooling_w) + 2 * padding_w_) / stride_w_ + 1));
        if (!output_w || !output_h) {
            LOG(ERROR) << "The output size of tensor " << i << "th"
                       << " in the max pooling layer is less than zero";
            return InferStatus::kInferFailedOutputSizeError;
        }
        const std::shared_ptr<ftensor>& output_data = outputs.at(i);
        if (output_data != nullptr && !output_data->empty()) {
            if (output_data->rows() != output_h ||
                output_data->cols() != output_w) {
                LOG(ERROR) << "The output tensor array in the max pooling layer "
                              "has an incorrectly sized tensor "
                           << i << "th";
                return InferStatus::kInferFailedOutputSizeError;
            }
        }
    }
    for (uint32_t i = 0; i < batch; ++i) {
        const std::shared_ptr<Tensor<float>>& input_data = inputs.at(i);
        CHECK(input_data == nullptr || !input_data->empty())
                        << "The input tensor array in the max pooling layer has an "
                           "empty tensor "
                        << i << "th";
        const uint32_t input_height = input_data->rows();
        const uint32_t input_width = input_data->cols();
        const uint32_t input_padded_height = input_data->rows() + 2 * padding_h_;
        const uint32_t input_padded_width = input_data->cols() + 2 * padding_w_;

        const uint32_t input_channel = input_data->channels();

        const auto output_h = uint32_t(std::floor((int(input_padded_height) - int(pooling_h)) / stride_h_ + 1));
        const auto output_w = uint32_t(std::floor((int(input_padded_width) - int(pooling_w)) / stride_w_ + 1));
        std::shared_ptr<Tensor<float>> output_data = outputs.at(i);
        if (output_data == nullptr || output_data->empty()) {
            output_data = std::make_shared<Tensor<float>>(input_channel, output_h, output_w);
            outputs.at(i) = output_data;
        }
        CHECK(output_data->rows() == output_h && output_data->cols() == output_w &&
              output_data->channels() == input_channel)
                        << "The output tensor array in the max pooling layer "
                           "has an incorrectly sized tensor "
                        << i << "th";
        // 开始计算逻辑
        for (uint32_t ic = 0; ic < input_channel; ic++) {
            const float* input_channel_data = input_data->slice(ic);
            float* output_channel_data = output_data->slice(ic);
            // 遍历输入tensor
            for (uint32_t row = 0; row < input_padded_height - pooling_h + 1; row += stride_h_) {
                int output_row = int (row / stride_h_);
                for (uint32_t col = 0; col < input_padded_width - pooling_w + 1; col += stride_w_) {
                    int output_col = int(col / stride_w_);
                    float max_value = std::numeric_limits<float>::lowest();
                    // 循环pooling核大小
                    for (uint32_t h = 0; h < pooling_h; ++h) {
                        for (uint32_t w = 0; w < pooling_w; ++w) {
                            uint32_t current_row = row + h - padding_h_;
                            uint32_t current_col = col + w - padding_w_;
                            float current_value = std::numeric_limits<float>::lowest();  // Assume padding with lowest value
                            if (current_row >= 0 && current_row < input_height && current_col >= 0 && current_col < input_width) {
                                // Convert 2D index to 1D index for the flattened array access
                                current_value = input_channel_data[current_row * input_width + current_col];
                            }
                            max_value = std::max(max_value, current_value);
                        }
                    }
                    output_channel_data[output_row * output_w + output_col] = max_value;
                }
            }
        }
    }
    return InferStatus::kInferSuccess;
}
ParseParameterAttrStatus MaxPoolingLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                      std::shared_ptr<Layer> &maxpooling_layer) {
    CHECK(op != nullptr) << "MaxPooling get instance failed, operator is nullptr";
    const std::map<std::string, std::shared_ptr<RuntimeParameter>>& params = op->params;
    if (params.find("stride") == params.end()) {
        LOG(ERROR) << "Can not find the stride parameter";
        return ParseParameterAttrStatus::kParameterMissingStride;
    }
    auto stride = std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("stride"));
    if (!stride) {
        LOG(ERROR) << "Can not find the stride parameter";
        return ParseParameterAttrStatus::kParameterMissingStride;
    }
    if (params.find("padding") == params.end()) {
        LOG(ERROR) << "Can not find the padding parameter";
        return ParseParameterAttrStatus::kParameterMissingPadding;
    }

    auto padding =
            std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("padding"));
    if (!padding) {
        LOG(ERROR) << "Can not find the padding parameter";
        return ParseParameterAttrStatus::kParameterMissingPadding;
    }

    if (params.find("kernel_size") == params.end()) {
        LOG(ERROR) << "Can not find the kernel size parameter";
        return ParseParameterAttrStatus::kParameterMissingKernel;
    }

    auto kernel_size = std::dynamic_pointer_cast<RuntimeParameterIntArray>(
            params.at("kernel_size"));
    if (!kernel_size) {
        LOG(ERROR) << "Can not find the kernel size parameter";
        return ParseParameterAttrStatus::kParameterMissingKernel;
    }
    const auto& padding_values = padding->value;
    const auto& stride_values = stride->value;
    const auto& kernel_values = kernel_size->value;
    const uint32_t dims = 2;
    if (padding_values.size() != dims) {
        LOG(ERROR) << "Can not find the right padding parameter";
        return ParseParameterAttrStatus::kParameterMissingPadding;
    }

    if (stride_values.size() != dims) {
        LOG(ERROR) << "Can not find the right stride parameter";
        return ParseParameterAttrStatus::kParameterMissingStride;
    }

    if (kernel_values.size() != dims) {
        LOG(ERROR) << "Can not find the right kernel size parameter";
        return ParseParameterAttrStatus::kParameterMissingKernel;
    }

    maxpooling_layer = std::make_shared<MaxPoolingLayer>(
            padding_values.at(0), padding_values.at(1), kernel_values.at(0),
            kernel_values.at(1), stride_values.at(0), stride_values.at(1));

    return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

// 注册算子
LayerRegistererWrapper kMaxPoolingGetInstance("nn.MaxPool2d", MaxPoolingLayer::GetInstance);
}