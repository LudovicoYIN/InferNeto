//
// Created by fss on 23-7-21.
//

#include "node/abstract/node_factory.hpp"
#include <gtest/gtest.h>
#include <vector>

using namespace infer_neto;

TEST(test_registry, create_layer_poolingforward) {
  std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
  op->type = "nn.MaxPool2d";
  std::vector<int> strides{2, 2};
  std::shared_ptr<RuntimeParameter> stride_param = std::make_shared<RuntimeParameterIntArray>(strides);
  op->params.insert({"stride", stride_param});

  std::vector<int> kernel{2, 2};
  std::shared_ptr<RuntimeParameter> kernel_param = std::make_shared<RuntimeParameterIntArray>(strides);
  op->params.insert({"kernel_size", kernel_param});

  std::vector<int> paddings{0, 0};
  std::shared_ptr<RuntimeParameter> padding_param = std::make_shared<RuntimeParameterIntArray>(paddings);
  op->params.insert({"padding", padding_param});

  std::shared_ptr<Layer> layer;
  layer = LayerRegisterer::CreateLayer(op);
  ASSERT_NE(layer, nullptr);
}

TEST(test_registry, create_layer_poolingforward_1) {
  std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
  op->type = "nn.MaxPool2d";
  std::vector<int> strides{2, 2};
  std::shared_ptr<RuntimeParameter> stride_param = std::make_shared<RuntimeParameterIntArray>(strides);
  op->params.insert({"stride", stride_param});

  std::vector<int> kernel{2, 2};
  std::shared_ptr<RuntimeParameter> kernel_param = std::make_shared<RuntimeParameterIntArray>(strides);
  op->params.insert({"kernel_size", kernel_param});

  std::vector<int> paddings{1, 1};
  std::shared_ptr<RuntimeParameter> padding_param = std::make_shared<RuntimeParameterIntArray>(paddings);
  op->params.insert({"padding", padding_param});

  std::shared_ptr<Layer> layer;
  layer = LayerRegisterer::CreateLayer(op);
  ASSERT_NE(layer, nullptr);

  sftensor tensor = std::make_shared<ftensor>(1, 4, 4);
  for(uint32_t i = 0; i < tensor->size(); i++) {
      tensor->index(i) = float(i);
  }
  tensor->Show();
  std::vector<sftensor> inputs(1);
  inputs.at(0) = tensor;
  std::vector<sftensor> outputs(1);
  layer->Forward(inputs, outputs);

  ASSERT_EQ(outputs.size(), 1);
  outputs.front()->Show();
}

