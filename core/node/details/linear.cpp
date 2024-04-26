//
// Created by hanke on 2024/4/26.
//

#include <iostream>
#include "linear.hpp"
#include "node/abstract/node_factory.hpp"

namespace infer_neto {


InferStatus LinearLayer::Forward(
        const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
        std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
    if (inputs.empty()) {
        LOG(ERROR) << "The input tensor array in the linear layer is empty";
        return InferStatus::kInferFailedInputEmpty;
    }

    if (inputs.size() != outputs.size()) {
        LOG(ERROR) << "The input and output tensor array size of linear layer do "
                      "not match";
        return InferStatus::kInferFailedInputOutSizeMatchError;
    }

    if (this->weights_.empty()) {
        LOG(ERROR) << "The weight tensor in the linear layer is empty";
        return InferStatus::kInferFailedWeightParameterError;
    } else {
        if (this->use_bias_ && this->weights_.size() != this->bias_.size()) {
            LOG(ERROR) << "The size of the weight and bias tensor do not match";
            return InferStatus::kInferFailedBiasParameterError;
        }
    }

    if (weights_.size() != 1) {
        LOG(ERROR) << "Need one weight tensor in the linear layer";
        return InferStatus::kInferFailedWeightParameterError;
    }

    if (use_bias_ && this->bias_.size() != 1) {
        LOG(ERROR) << "Need one bias tensor in the linear layer";
        return InferStatus::kInferFailedBiasParameterError;
    }

    uint32_t batch = inputs.size();
    const std::shared_ptr<Tensor<float>>& weight = weights_.front();
    for (auto i : weight->shapes()) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    for (auto i : inputs.at(0)->shapes()) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    for (auto i : outputs.at(0)->shapes()) {
        std::cout << i << " ";
    }


    for (uint32_t i = 0; i < batch; ++i) {
        const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
        CHECK(input != nullptr && !input->empty())
                        << "The input tensor array in the linear layer has an empty tensor "
                        << i << " th";
        const std::vector<uint32_t>& input_shapes = input->shapes();

        const uint32_t feature_dims = input_shapes.at(1);
        const uint32_t in_features = input_shapes.at(2);
        CHECK(weight->rows() == out_features_)
                        << "The row of weight tensor should be same to output_features_";
        CHECK(weight->cols() == in_features && in_features == in_features_)
                        << "The col of weight tensor should be same to input_features_";



        std::shared_ptr<Tensor<float>> output = outputs.at(i);
        if (output == nullptr || output->empty()) {
            output = std::make_shared<Tensor<float>>(1, out_features_, feature_dims);
            outputs.at(i) = output;
        }
        CHECK(output->channels() == 1 && output->rows() == feature_dims &&
              output->cols() == out_features_)
                        << "The row of output tensor should be same to feature_dims_ and the "
                           "col of output tensor should be same to output_features_ "
                        << i << " th";
        const auto& output_raw_shapes = output->raw_shapes();
        if (output_raw_shapes.size() == 2) {
            CHECK(output_raw_shapes.at(0) == feature_dims &&
                  output_raw_shapes.at(1) == out_features_);
        }
        if (output_raw_shapes.size() == 1) {
            CHECK(output_raw_shapes.at(0) == out_features_);
        }

        Tensor<float> result = input->Gemm(*weight);
        if (use_bias_) {
            CHECK(!this->bias_.empty() && this->bias_.size() == 1)
                            << "The bias tensor is empty, but use_bias is true";

            const auto& bias_data = bias_.front();
            CHECK(!bias_data->empty() && bias_data->cols() == out_features_)
                            << "The col of bias tensor is not same to output_features_";
            result.Add(*bias_data);
        }
        *output = std::move(result);
    }
    return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus LinearLayer::GetInstance(
        const std::shared_ptr<RuntimeOperator>& op,
        std::shared_ptr<Layer>& linear_layer) {
    CHECK(op != nullptr) << "Linear operator is nullptr";
    const auto& params = op->params;
    if (params.find("bias") == params.end()) {
        LOG(ERROR) << "Can not find the use bias parameter";
        return ParseParameterAttrStatus::kParameterMissingUseBias;
    }
    auto use_bias_param =
            std::dynamic_pointer_cast<RuntimeParameterBool>(params.at("bias"));
    if (use_bias_param == nullptr) {
        LOG(ERROR) << "Can not find the use bias parameter";
        return ParseParameterAttrStatus::kParameterMissingUseBias;
    }

    const auto& attr = op->attribute;
    CHECK(!attr.empty()) << "Operator attributes is empty";

    if (attr.find("weight") == attr.end()) {
        LOG(ERROR) << "Can not find the weight parameter";
        return ParseParameterAttrStatus::kAttrMissingWeight;
    }

    if (use_bias_param->value) {
        if (attr.find("bias") == attr.end()) {
            LOG(ERROR) << "Can not find the bias parameter";
            return ParseParameterAttrStatus::kAttrMissingBias;
        }
    }

    const auto& weight = attr.at("weight");
    const auto& bias = attr.at("bias");
    const auto& shapes = weight->shape;
    if ((shapes.size() < 2)) {
        LOG(ERROR) << "The graph only support two dimension matrix multiply";
        return ParseParameterAttrStatus::kAttrMissingOutFeatures;
    }

    int32_t out_features = shapes.at(0);
    int32_t in_features = shapes.at(1);
    const bool use_bias = use_bias_param->value;

    linear_layer = std::make_shared<LinearLayer>(in_features, out_features, use_bias);
    if (use_bias) {
        linear_layer->set_bias(bias->get<float>());
    }

    // load weights
    linear_layer->set_weights(weight->get<float>());
    return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kLinearCreateInstance("nn.Linear",
                                             LinearLayer::GetInstance);
}