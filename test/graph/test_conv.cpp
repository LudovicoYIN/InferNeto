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
    std::vector<sftensor> inputs(batch_size);
    std::vector<sftensor> outputs(batch_size);

    const uint32_t in_channel = 2;
    for (uint32_t i = 0; i < batch_size; ++i) {
    sftensor input = std::make_shared<ftensor>(in_channel, 4, 4);
    float * data = input->slice(0);
    for (int t = 1; t <= 16; t++) {
        data[t] = float (t);
    }
    float * data2 = input->slice(1);
      for (int t = 1; t <= 16; t++) {
          data2[t] = float (t);
      }
    inputs.at(i) = input;
    }
    const uint32_t kernel_h = 3;
    const uint32_t kernel_w = 3;
    const uint32_t stride_h = 1;
    const uint32_t stride_w = 1;
    const uint32_t kernel_count = 2;
    std::vector<sftensor> weights;
    for (uint32_t i = 0; i < kernel_count; ++i) {
    sftensor kernel = std::make_shared<ftensor>(in_channel, kernel_h, kernel_w);
    float * data3 = kernel->slice(0);

    for (int t = 1; t <= 9; t++) {
        data3[t] = float (t);
    }
    float * data4 = kernel->slice(1);
    for (int t = 1; t <= 9; t++) {
        data4[t] = float (t);
    }
    weights.push_back(kernel);
  }
  ConvolutionLayer conv_layer(kernel_count, in_channel, kernel_h, kernel_w, 0,
                              0, stride_h, stride_w, 1, false);
  conv_layer.set_weights(weights);
  conv_layer.Forward(inputs, outputs);
  outputs.at(0)->Show();
}