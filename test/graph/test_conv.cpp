//
// Created by fss on 23-7-22.
//
#include "node/abstract/node_factory.hpp"
#include "node/details/convolution.hpp"
#include <gtest/gtest.h>
#include <vector>

using namespace infer_neto;

TEST(test_registry, create_layer_convforward) {
    const uint32_t batch_size = 1;
    std::vector<std::shared_ptr<Tensor<float>>> inputs(batch_size);
    std::vector<std::shared_ptr<Tensor<float>>> outputs(batch_size);

    const uint32_t in_channel = 2;
    const uint32_t input_height = 4;
    const uint32_t input_width = 4;
    for (uint32_t i = 0; i < batch_size; ++i) {
        auto input = std::make_shared<Tensor<float>>(in_channel, input_height, input_width);
        std::vector<float> data = {
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        };
        input->Fill(data);
        inputs[i] = input;
    }

    const uint32_t kernel_h = 3;
    const uint32_t kernel_w = 3;
    const uint32_t stride_h = 1;
    const uint32_t stride_w = 1;
    const uint32_t kernel_count = 2;
    std::vector<std::shared_ptr<Tensor<float>>> weights;
    for (uint32_t i = 0; i < kernel_count; ++i) {
        auto kernel = std::make_shared<Tensor<float>>(in_channel, kernel_h, kernel_w);
        std::vector<float> kernel_data = {
                1, 2, 3, 4, 5, 6, 7, 8, 9,
                1, 2, 3, 4, 5, 6, 7, 8, 9
        };
        kernel->Fill(kernel_data);
        weights.push_back(kernel);
    }

    ConvolutionLayer conv_layer(kernel_count, in_channel, kernel_h, kernel_w, 0, 0, stride_h, stride_w, 1, false);
    conv_layer.set_weights(weights);
    conv_layer.Forward(inputs, outputs);

    for (auto& output : outputs) {
        if (output) output->Show();
        else std::cout << "Output tensor is nullptr." << std::endl;
    }
}