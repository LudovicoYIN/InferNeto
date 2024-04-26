//
// Created by fss on 23-7-22.
//
#include "node/abstract/node_factory.hpp"
#include "gtest/gtest.h"
#include "node/details/linear.hpp"

TEST(LinearLayerTest, ForwardPass) {
    using namespace infer_neto;
    const uint32_t batch_size = 1;
    const int32_t in_features = 16; // Assuming these are the dimensions expected by your layer
    const int32_t out_features = 8;
    const bool use_bias = true;

    std::vector<std::shared_ptr<Tensor<float>>> inputs(batch_size);
    std::vector<std::shared_ptr<Tensor<float>>> outputs(batch_size);

    for (uint32_t i = 0; i < batch_size; ++i) {
        auto input = std::make_shared<Tensor<float>>(1, in_features); // 1D input
        input->Rand();  // Random initialization
        inputs[i] = input;
    }

    std::vector<std::shared_ptr<Tensor<float>>> weights;
    auto weight = std::make_shared<Tensor<float>>(in_features, out_features);
    weight->Rand();
    weights.push_back(weight);

    std::vector<std::shared_ptr<Tensor<float>>> biases;
    if (use_bias) {
        auto bias = std::make_shared<Tensor<float>>(1, out_features); // 1D tensor
        bias->Rand();
        biases.push_back(bias);
    }

    LinearLayer linearLayer(in_features, out_features, use_bias);
    // Dimension check before setting weights
    if (weight->rows() != in_features || weight->cols() != out_features) {
        std::cerr << "Weight dimensions mismatch. Expected rows: " << in_features
                  << ", cols: " << out_features << std::endl;
        ASSERT_TRUE(false);
    }
    linearLayer.set_weights(weights);

    if (use_bias && biases.front()->cols() != out_features) {
        std::cerr << "Bias dimensions mismatch. Expected size: " << out_features << std::endl;
        ASSERT_TRUE(false);
    }
    if (use_bias) {
        linearLayer.set_bias(biases);
    }

    ASSERT_EQ(linearLayer.Forward(inputs, outputs), InferStatus::kInferSuccess);

    for (auto& output : outputs) {
        if (output) {
            output->Show();
        } else {
            std::cout << "Output tensor is nullptr." << std::endl;
        }
    }
}