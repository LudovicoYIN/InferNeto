//
// Created by hanke on 2024/4/3.
//
#include <glog/logging.h>
#include <valarray>
#include "data/cpu/tensor.hpp"
#include "data/cpu/tensor_util.hpp"

namespace infer_neto {
    bool TensorIsSame(const std::shared_ptr<Tensor<float>>& a,
                      const std::shared_ptr<Tensor<float>>& b, float threshold) {
        CHECK(a != nullptr);
        CHECK(b != nullptr);
        if (a->shapes() != b->shapes()) {
            return false;
        }
        float* a_data = a->data().get();
        float* b_data = b->data().get();

        for (uint32_t i = 0; i < a->size(); i++) {
            if (std::abs(a_data[i] - b_data[i]) > threshold) return false;
        }
        return true;
    }

    void TensorElementAdd(const std::shared_ptr<Tensor<float>>& tensor1,
                          const std::shared_ptr<Tensor<float>>& tensor2,
                          const std::shared_ptr<Tensor<float>>& output_tensor) {
        CHECK(tensor1 != nullptr && tensor2 != nullptr && output_tensor != nullptr);
        CHECK(tensor1->shapes() == tensor2->shapes()) << "Input tensors must have the same shapes.";
        CHECK(tensor1->shapes() == output_tensor->shapes()) << "Output tensor must have the same shape as input tensors.";

        const float* data1 = tensor1->data().get();
        const float* data2 = tensor2->data().get();
        float* output_data = output_tensor->data().get();

        for (size_t i = 0; i < tensor1->size(); ++i) {
            output_data[i] = data1[i] + data2[i];
        }
    }

    void TensorElementMultiply(
            const std::shared_ptr<Tensor<float>>& tensor1,
            const std::shared_ptr<Tensor<float>>& tensor2,
            const std::shared_ptr<Tensor<float>>& output_tensor) {
        CHECK(tensor1 != nullptr && tensor2 != nullptr && output_tensor != nullptr);
        CHECK(tensor1->shapes() == tensor2->shapes()) << "Input tensors must have the same shapes.";
        CHECK(tensor1->shapes() == output_tensor->shapes()) << "Output tensor must have the same shape as input tensors.";

        const float* data1 = tensor1->data().get();
        const float* data2 = tensor2->data().get();
        float* output_data = output_tensor->data().get();

        for (size_t i = 0; i < tensor1->size(); ++i) {
            output_data[i] = data1[i] * data2[i];
        }
    }

    std::shared_ptr<Tensor<float>> TensorElementSin(
            const std::shared_ptr<Tensor<float>>& tensor) {
        CHECK(tensor != nullptr);
        sftensor output_tensor = TensorCreate(tensor->shapes());
        const float* data = tensor->data().get();
        float* output_data = output_tensor->data().get();

        for (size_t i = 0; i < tensor->size(); ++i) {
            output_data[i] = std::sin(data[i]);
        }
        return output_tensor;
    }

    std::shared_ptr<Tensor<float>> TensorElementAdd(
            const std::shared_ptr<Tensor<float>>& tensor1,
            const std::shared_ptr<Tensor<float>>& tensor2) {
        CHECK(tensor1 != nullptr && tensor2 != nullptr);
        CHECK(tensor1->shapes() == tensor2->shapes()) << "Input tensors must have the same shapes.";
        sftensor output_tensor = TensorCreate(tensor1->shapes());
        const float* data1 = tensor1->data().get();
        const float* data2 = tensor2->data().get();
        float* output_data = output_tensor->data().get();

        for (size_t i = 0; i < tensor1->size(); ++i) {
            output_data[i] = data1[i] + data2[i];
        }
        return output_tensor;
    }

    std::shared_ptr<Tensor<float>> TensorElementMultiply(
            const std::shared_ptr<Tensor<float>>& tensor1,
            const std::shared_ptr<Tensor<float>>& tensor2) {
        CHECK(tensor1 != nullptr && tensor2 != nullptr);
        CHECK(tensor1->shapes() == tensor2->shapes()) << "Input tensors must have the same shapes.";
        sftensor output_tensor = TensorCreate(tensor1->shapes());
        const float* data1 = tensor1->data().get();
        const float* data2 = tensor2->data().get();
        float* output_data = output_tensor->data().get();

        for (size_t i = 0; i < tensor1->size(); ++i) {
            output_data[i] = data1[i] * data2[i];
        }
        return output_tensor;
    }

    std::shared_ptr<Tensor<float>> TensorCreate(uint32_t channels, uint32_t rows,
                                                uint32_t cols) {
        return std::make_shared<Tensor<float>>(channels, rows, cols);
    }

    std::shared_ptr<Tensor<float>> TensorCreate(uint32_t rows, uint32_t cols) {
        return std::make_shared<Tensor<float>>(1, rows, cols);
    }

    std::shared_ptr<Tensor<float>> TensorCreate(uint32_t size) {
        return std::make_shared<Tensor<float>>(1, 1, size);
    }

    std::shared_ptr<Tensor<float>> TensorCreate(
            const std::vector<uint32_t>& shapes) {
        CHECK(!shapes.empty() && shapes.size() <= 3);
        if (shapes.size() == 1) {
            return std::make_shared<Tensor<float>>(shapes.at(0));
        } else if (shapes.size() == 2) {
            return std::make_shared<Tensor<float>>(shapes.at(0), shapes.at(1));
        } else {
            return std::make_shared<Tensor<float>>(shapes.at(0), shapes.at(1),
                                                   shapes.at(2));
        }
    }

    std::shared_ptr<Tensor<float>> TensorClone(
            const std::shared_ptr<Tensor<float>>& tensor) {
        return std::make_shared<Tensor<float>>(*tensor);
    }
}