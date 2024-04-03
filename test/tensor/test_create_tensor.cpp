//
// Created by hanke on 2024/4/2.
//
//
// Created by hanke on 2024/4/2.
//
#include "tensor.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

TEST(test_tensor, tensor_init1D) {
    using namespace infer_neto;
    Tensor<float> f1({4});
    f1.fill(1.f);
    const auto &raw_shapes = f1.shape();
    LOG(INFO) << "-----------------------Tensor1D-----------------------";
    LOG(INFO) << "raw shapes size: " << raw_shapes.size();
    const uint32_t size = raw_shapes.at(0);
    LOG(INFO) << "data numbers: " << size;
    f1.print();
}

TEST(test_tensor, tensor_init2D) {
    using namespace infer_neto;
    Tensor<float> f1({4, 4});
    f1.fill(1.f);

    const auto &raw_shapes = f1.shape();
    LOG(INFO) << "-----------------------Tensor2D-----------------------";
    LOG(INFO) << "raw shapes size: " << raw_shapes.size();
    const uint32_t rows = raw_shapes.at(0);
    const uint32_t cols = raw_shapes.at(1);

    LOG(INFO) << "data rows: " << rows;
    LOG(INFO) << "data cols: " << cols;
    f1.print();
}

TEST(test_tensor, tensor_init3D_1) {
    using namespace infer_neto;
    Tensor<float> f1({2, 2, 3});
    f1.fill(1.f);

    const auto &raw_shapes = f1.shape();
    LOG(INFO) << "-----------------------Tensor3D 1-----------------------";
    LOG(INFO) << "raw shapes size: " << raw_shapes.size();
    const uint32_t size = raw_shapes.at(0);
    LOG(INFO) << "data numbers: " << size;
    f1.print();
}

TEST(test_tensor, tensor_init3D_3) {
    using namespace infer_neto;
    Tensor<float> f1({2, 3, 4});
    f1.fill(1.f);

    const auto &raw_shapes = f1.shape();
    LOG(INFO) << "-----------------------Tensor3D 3-----------------------";
    LOG(INFO) << "raw shapes size: " << raw_shapes.size();
    const uint32_t channels = raw_shapes.at(0);
    const uint32_t rows = raw_shapes.at(1);
    const uint32_t cols = raw_shapes.at(2);

    LOG(INFO) << "data channels: " << channels;
    LOG(INFO) << "data rows: " << rows;
    LOG(INFO) << "data cols: " << cols;
    f1.print();
}

TEST(test_tensor, tensor_init3D_2) {
    using namespace infer_neto;
    Tensor<float> f1({1, 2, 3});
    f1.fill(1.f);

    const auto &raw_shapes = f1.shape();
    LOG(INFO) << "-----------------------Tensor3D 2-----------------------";
    LOG(INFO) << "raw shapes size: " << raw_shapes.size();
    const uint32_t rows = raw_shapes.at(0);
    const uint32_t cols = raw_shapes.at(1);

    LOG(INFO) << "data rows: " << rows;
    LOG(INFO) << "data cols: " << cols;
    f1.print();
}

// 测试复制构造函数
TEST(test_tensor, CopyConstructor) {
    infer_neto::Tensor<float> original({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    infer_neto::Tensor<float> copy = original;  // 使用复制构造函数

    // 检查形状是否相同
    ASSERT_EQ(copy.shape(), original.shape());

    // 检查数据是否相同
    for (uint32_t i = 0; i < original.channels(); ++i) {
        for (uint32_t j = 0; i < original.rows(); ++i) {
            for (uint32_t k = 0; i < original.cols(); ++i) {
                EXPECT_FLOAT_EQ(copy.at({i, j, k}), original.at({i, j, k}));
            }
        }
    }
}

// 测试移动构造函数
TEST(test_tensor, MoveConstructor) {
    infer_neto::Tensor<float> original({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    infer_neto::Tensor<float> moved = std::move(original);  // 使用移动构造函数

    // 检查原始Tensor是否为空
    ASSERT_TRUE(original.empty());

    // 检查新Tensor的形状
    ASSERT_EQ(moved.shape().size(), 2);
    ASSERT_EQ(moved.shape()[0], 2);
    ASSERT_EQ(moved.shape()[1], 3);

    // 检查数据
    std::vector<float> expectedValues = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    // 检查数据是否相同
    for (uint32_t i = 0; i < original.channels(); ++i) {
        for (uint32_t j = 0; i < original.rows(); ++i) {
            for (uint32_t k = 0; i < original.cols(); ++i) {
                EXPECT_FLOAT_EQ(moved.at({i, j, k}), expectedValues[i * j * k]);
            }
        }
    }
}

// 测试复制赋值操作符
TEST(test_tensor, CopyAssignment) {
    infer_neto::Tensor<float> original({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    infer_neto::Tensor<float> copy;
    copy = original;  // 使用复制赋值操作符

    // 检查形状是否相同
    ASSERT_EQ(copy.shape(), original.shape());


    // 检查数据是否相同
    for (uint32_t i = 0; i < original.channels(); ++i) {
        for (uint32_t j = 0; i < original.rows(); ++i) {
            for (uint32_t k = 0; i < original.cols(); ++i) {
                EXPECT_FLOAT_EQ(copy.at({i, j, k}), original.at({i, j, k}));
            }
        }
    }
}

// 测试移动赋值操作符
TEST(test_tensor, MoveAssignment) {
    infer_neto::Tensor<float> original({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    infer_neto::Tensor<float> moved;
    moved = std::move(original);  // 使用移动赋值操作符

    // 检查原始Tensor是否为空
    ASSERT_TRUE(original.empty());

    // 检查新Tensor的形状
    ASSERT_EQ(moved.shape().size(), 2);
    ASSERT_EQ(moved.shape()[0], 2);
    ASSERT_EQ(moved.shape()[1], 3);

    // 检查数据
    std::vector<float> expectedValues = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    // 检查数据是否相同
    for (uint32_t i = 0; i < original.channels(); ++i) {
        for (uint32_t j = 0; i < original.rows(); ++i) {
            for (uint32_t k = 0; i < original.cols(); ++i) {
                EXPECT_FLOAT_EQ(moved.at({i, j, k}), expectedValues[i * j * k]);
            }
        }
    }
}

