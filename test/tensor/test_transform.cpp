//
// Created by fss on 23-6-4.
//
#include "data/cpu/tensor.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

void MinusOne(float& value) {
    value -= 1.f;
}
TEST(test_transform, transform1) {
  using namespace infer_neto;
  Tensor<float> f1({2, 3, 4});
  f1.random();
  f1.print();
  f1.transform(MinusOne);
  f1.print();
}