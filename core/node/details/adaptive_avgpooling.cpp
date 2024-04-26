//
// Created by hanke on 2024/4/26.
//

#include <cmath>
#include "adaptive_avgpooling.hpp"
#include "node/abstract/node_factory.hpp"

namespace infer_neto {
InferStatus AdaptiveAveragePoolingLayer::Forward(
        const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
        std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
    if (inputs.empty()) {
        LOG(ERROR)
                << "The input tensor array in the adaptive pooling layer is empty";
        return InferStatus::kInferFailedInputEmpty;
    }

    if (inputs.size() != outputs.size()) {
        LOG(ERROR) << "The input and output tensor array size of the adaptive "
                      "pooling layer "
                      "do not match";
        return InferStatus::kInferFailedInputOutSizeMatchError;
    }

    const uint32_t batch = inputs.size();
    for (uint32_t i = 0; i < batch; ++i) {
        const std::shared_ptr<ftensor>& input_data = inputs.at(i);
        const std::shared_ptr<ftensor>& output_data = outputs.at(i);
        if (input_data == nullptr || input_data->empty()) {
            LOG(ERROR) << "The input tensor array in the adaptive pooling layer has "
                          "an empty tensor "
                       << i << "th";
            return InferStatus::kInferFailedInputEmpty;
        }
        if (output_data != nullptr && !output_data->empty()) {
            if (output_data->rows() != output_h_ ||
                output_data->cols() != output_w_) {
                LOG(ERROR) << "The output tensor array in the adaptive pooling layer "
                              "has an incorrectly sized tensor "
                           << i << "th";
                return InferStatus::kInferFailedOutputSizeError;
            }
        }
    }

    for (uint32_t i = 0; i < batch; ++i) {
        const std::shared_ptr<Tensor<float>>& input_data = inputs.at(i);
        CHECK(input_data != nullptr && !input_data->empty())
                        << "The input tensor array in the adaptive pooling layer has an empty "
                           "tensor "
                        << i << "th";

        const uint32_t input_h = input_data->rows();
        const uint32_t input_w = input_data->cols();
        const uint32_t input_c = input_data->channels();
        const uint32_t stride_h = uint32_t(std::floor(input_h / output_h_));
        const uint32_t stride_w = uint32_t(std::floor(input_w / output_w_));
        CHECK(stride_w > 0 && stride_h > 0)
                        << "The stride parameter is set incorrectly. It must always be greater "
                           "than 0";

        const uint32_t pooling_h = (int)input_h - (int(output_h_) - 1) * int(stride_h);
        const uint32_t pooling_w = (int)input_w - (int(output_w_) - 1) * int(stride_w);
        CHECK(pooling_w > 0 && pooling_h > 0)
                        << "The pooling parameter is set incorrectly. It must always be "
                           "greater than 0";

        std::shared_ptr<Tensor<float>> output_data = outputs.at(i);
        if (output_data == nullptr || output_data->empty()) {
            DLOG(ERROR) << "The output tensor array in the adaptive pooling layer "
                           "has an empty tensor "
                        << i << "th";
            output_data =
                    std::make_shared<Tensor<float>>(input_c, output_h_, output_w_);
            outputs.at(i) = output_data;
        }

        CHECK(output_data->rows() == output_h_ &&
              output_data->cols() == output_w_ &&
              output_data->channels() == input_c)
                        << "The output tensor array in the adaptive pooling layer has an "
                           "incorrectly sized tensor "
                        << i << "th";

        const uint32_t pooling_size = pooling_h * pooling_w;
        for (uint32_t ic = 0; ic < input_c; ++ic) {
            const float*  input_channel = input_data->slice(ic);
            float* output_channel = output_data->slice(ic);
            for (uint32_t row = 0; row < input_h - pooling_h + 1; row += stride_h) {
                int output_row = int(row / stride_h);
                for (uint32_t col = 0; col < input_w - pooling_w + 1; col += stride_w) {
                    int output_col = int(col / stride_w);
                    float mean_value = 0.f;
                    for (uint32_t w = 0; w < pooling_w; ++w) {
                        for (uint32_t h = 0; h < pooling_h; ++h) {
                            uint32_t current_row = row + h;
                            uint32_t current_col = col + w;
                            float current_value = input_channel[current_row * input_w + current_col];
                            mean_value = mean_value + current_value;
                        }
                    }
                    output_channel[output_row * output_w_ + output_col] = mean_value / float(pooling_size);
                }
            }
        }
    }
    return InferStatus::kInferSuccess;
}
ParseParameterAttrStatus AdaptiveAveragePoolingLayer::CreateInstance(
        const std::shared_ptr<RuntimeOperator>& op,
        std::shared_ptr<Layer>& avg_layer) {
    CHECK(op != nullptr) << "Adaptive pooling operator is nullptr";
    const auto& params = op->params;
    CHECK(!params.empty()) << "Operator parameter is empty";

    auto output_hw = std::dynamic_pointer_cast<RuntimeParameterIntArray>(
            params.at("output_size"));
    if (!output_hw) {
        LOG(ERROR) << "Can not find the output size parameter";
        return ParseParameterAttrStatus::kParameterMissingOutHW;
    }

    const auto& output_hw_arr = output_hw->value;
    if (output_hw_arr.size() != 2) {
        LOG(ERROR) << "Can not find the output size parameter";
        return ParseParameterAttrStatus::kParameterMissingOutHW;
    }
    avg_layer = std::make_shared<AdaptiveAveragePoolingLayer>(
            output_hw_arr.at(0), output_hw_arr.at(1));
    return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kAdaptiveAvgpoolingCreateInstance(
        "nn.AdaptiveAvgPool2d", AdaptiveAveragePoolingLayer::CreateInstance);
}