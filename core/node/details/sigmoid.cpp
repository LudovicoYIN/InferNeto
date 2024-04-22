// MIT License
// Copyright (c) 2022 - 傅莘莘
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Created by fss on 22-11-18.
#include "node/abstract/node_factory.hpp"
#include "sigmoid.hpp"
#include <cmath>

namespace infer_neto {
InferStatus SigmoidLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                  std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the sigmoid layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }
  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the sigmoid layer do "
                  "not match";
    return InferStatus::kInferFailedInputOutSizeMatchError;
  }

  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const sftensor &input_data = inputs.at(i);
    const sftensor &output_data = outputs.at(i);
    if (input_data == nullptr || input_data->empty()) {
      LOG(ERROR)
          << "The input tensor array in the sigmoid layer has an empty tensor "
          << i << " th";
      return InferStatus::kInferFailedInputEmpty;
    }
    if (output_data != nullptr && !output_data->empty()) {
      if (input_data->shapes() != output_data->shapes()) {
        LOG(ERROR) << "The input and output tensor shapes of the sigmoid "
                      "layer do not match "
                   << i << " th";
        return InferStatus::kInferFailedInputOutSizeMatchError;
      }
    }
  }

  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>> &input = inputs.at(i);
    CHECK(input == nullptr || !input->empty())
            << "The input tensor array in the sigmoid layer has an empty tensor " << i
            << " th";

    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      DLOG(ERROR)
          << "The output tensor array in the sigmoid layer has an empty tensor "
          << i << " th";
      output = std::make_shared<Tensor<float>>(input->shapes());
      outputs.at(i) = output;
    }
    CHECK(output->shapes() == input->shapes())
            << "The input and output tensor shapes of the sigmoid layer do not match "
            << i << " th";
    for (uint32_t j = 0; j < input->size(); ++j) {
      float value = input->index(j);

      output->index(j) = 1 / (1.f + std::exp(-value));
    }
  }
  return InferStatus::kInferSuccess;
}
ParseParameterAttrStatus SigmoidLayer::GetInstance(
    const std::shared_ptr<RuntimeOperator> &op,
    std::shared_ptr<Layer> &sigmoid_layer) {
  CHECK(op != nullptr) << "Sigmoid operator is nullptr";
    sigmoid_layer = std::make_shared<SigmoidLayer>();
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

// 使用工具类注册算子
LayerRegistererWrapper kSigmoidGetInstance("nn.Sigmoid", SigmoidLayer::GetInstance);
}  // namespace kuiper_infer
