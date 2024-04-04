//
// Created by hanke on 2024/4/3.
//

#ifndef INFERNETO_TENSOR_UTIL_HPP
#define INFERNETO_TENSOR_UTIL_HPP
#include "data/cpu/tensor.hpp"
namespace infer_neto {
    std::shared_ptr<Tensor<float>> TensorCreate(uint32_t channels, uint32_t rows,
                                                uint32_t cols);


    std::shared_ptr<Tensor<float>> TensorCreate(uint32_t rows, uint32_t cols);


    std::shared_ptr<Tensor<float>> TensorCreate(uint32_t size);

    std::shared_ptr<Tensor<float>> TensorCreate(
            const std::vector<uint32_t>& shapes);
}

#endif //INFERNETO_TENSOR_UTIL_HPP
