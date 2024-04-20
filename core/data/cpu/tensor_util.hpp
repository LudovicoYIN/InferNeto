//
// Created by hanke on 2024/4/3.
//

#ifndef INFERNETO_TENSOR_UTIL_HPP
#define INFERNETO_TENSOR_UTIL_HPP
#include "data/cpu/tensor.hpp"
namespace infer_neto {
/**
 * 比较tensor的值是否相同
 * @param a 输入张量1
 * @param b 输入张量2
 * @param threshold 张量之间差距的阈值
 * @return 比较结果
 */
    bool TensorIsSame(const std::shared_ptr<Tensor<float>>& a,
                      const std::shared_ptr<Tensor<float>>& b,
                      float threshold = 1e-5f);

/**
 * 张量相加
 * @param tensor1 输入张量1
 * @param tensor2 输入张量2
 * @return 张量相加的结果
 */
    std::shared_ptr<Tensor<float>> TensorElementAdd(
            const std::shared_ptr<Tensor<float>>& tensor1,
            const std::shared_ptr<Tensor<float>>& tensor2);

/**
 * 张量相加
 * @param tensor1 输入张量1
 * @param tensor2 输入张量2
 * @param output_tensor 输出张量
 */
    void TensorElementAdd(const std::shared_ptr<Tensor<float>>& tensor1,
                          const std::shared_ptr<Tensor<float>>& tensor2,
                          const std::shared_ptr<Tensor<float>>& output_tensor);

/**
 * 矩阵点乘
 * @param tensor1 输入张量1
 * @param tensor2 输入张量2
 * @param output_tensor 输出张量
 */
    void TensorElementMultiply(const std::shared_ptr<Tensor<float>>& tensor1,
                               const std::shared_ptr<Tensor<float>>& tensor2,
                               const std::shared_ptr<Tensor<float>>& output_tensor);

/**
 * 张量相乘
 * @param tensor1 输入张量1
 * @param tensor2 输入张量2
 * @return 张量相乘的结果
 */
    std::shared_ptr<Tensor<float>> TensorElementMultiply(
            const std::shared_ptr<Tensor<float>>& tensor1,
            const std::shared_ptr<Tensor<float>>& tensor2);

/**
 * 创建一个张量
 * @param channels 通道数量
 * @param rows 行数
 * @param cols 列数
 * @return 创建后的张量
 */
    std::shared_ptr<Tensor<float>> TensorCreate(uint32_t channels, uint32_t rows,
                                                uint32_t cols);

/**
 * 创建一个张量
 * @param shapes 张量的形状
 * @return 创建后的张量
 */
    std::shared_ptr<Tensor<float>> TensorCreate(
            const std::vector<uint32_t>& shapes);

/**
 * 返回一个深拷贝后的张量
 * @param 待Clone的张量
 * @return 新的张量
 */
    std::shared_ptr<Tensor<float>> TensorClone(
            const std::shared_ptr<Tensor<float>>& tensor);

}

#endif //INFERNETO_TENSOR_UTIL_HPP
