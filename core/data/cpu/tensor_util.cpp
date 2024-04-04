//
// Created by hanke on 2024/4/3.
//
#include <glog/logging.h>
#include "data/cpu/tensor.hpp"
#include "data/cpu/tensor_util.hpp"

namespace infer_neto {
std::shared_ptr<Tensor<float>> TensorCreate(uint32_t channels, uint32_t rows,
                                            uint32_t cols) {
    // 将 channels, rows, cols 封装在一个 vector 中
    std::vector<uint32_t> shape = {channels, rows, cols};

    // 使用构造好的 shape 向量创建 Tensor 实例
    // 注意，由于 Tensor<float> 的构造函数接受 std::vector<uint32_t> 作为第一个参数，
    // 您需要根据您的 Tensor 类构造函数的定义，确保提供适当的第二个参数
    // 这里示例假设您有一个接受 shape 但不需要初始数据向量的构造函数
    return std::make_shared<Tensor<float>>(shape);
}
std::shared_ptr<Tensor<float>> TensorCreate(uint32_t rows, uint32_t cols) {
    std::vector<uint32_t> shape = {rows, cols};
    return std::make_shared<Tensor<float>>(shape);
}
std::shared_ptr<Tensor<float>> TensorCreate(uint32_t size) {
    std::vector<uint32_t> shape = {size};
    return std::make_shared<Tensor<float>>(shape);
}

std::shared_ptr<Tensor<float>> TensorCreate(const std::vector<uint32_t> &shapes) {
    CHECK(shapes.size() == 3);
    return TensorCreate(shapes.at(0), shapes.at(1), shapes.at(2));
}
}