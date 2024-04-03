//
// Created by fss on 23-6-4.
//

#include "tensor.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>
TEST(test_tensor_values, tensor_values1) {
    using namespace infer_neto;
    Tensor<float> f1({2, 3, 4});
    f1.random();
    f1.print();

    LOG(INFO) << "Data in the first channel: ";
    f1.slice(0).print();
    LOG(INFO) << "Data in the (1,1,1): " << f1.at({1, 0, 0});
}