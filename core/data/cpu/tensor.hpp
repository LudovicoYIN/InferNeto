//
// Created by hanke on 2024/3/31.
//

#ifndef INFERNETO_TENSOR_H
#define INFERNETO_TENSOR_H

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

/**
 * @brief 多维张量类
 *
 * @tparam T 存储在张量中的元素类型
 */
namespace infer_neto {
template<typename T>
class Tensor {
public:
    Tensor();

    // 构造函数
    explicit Tensor(const std::vector<uint32_t>& shape, const std::vector<T>& data);
    explicit Tensor(const std::vector<uint32_t>& shape, T fillValue = T());

    // 析构函数
    ~Tensor() = default;

    Tensor(const Tensor& other); // 复制构造函数
    Tensor(Tensor&& other) noexcept; // 移动构造函数
    Tensor& operator=(const Tensor& other); // 复制赋值操作符
    Tensor& operator=(Tensor&& other) noexcept; // 移动赋值操作符

    // 获取shape
    const std::vector<uint32_t>& shape() const;

    // 获取总元素个数
    uint32_t size() const;

    // 基础数学运算
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    // 随机填充
    void random(T lower_bound = 0, T upper_bound = 1);

    // 随机填充
    void fill(T data = 0);

    // 随机填充
    void fill(const std::vector<T>& values);

    // 转置
    Tensor transpose() const;

    // 重塑
    void reshape(const std::vector<uint32_t> &newShape);

    // 基础打印函数
    void print() const;

    // 打印张量
    void show();

    // 按索引取值、修改值
    T& at(const std::vector<uint32_t>& indices);


    // 按索引取值，不可修改
    const T& at(const std::vector<uint32_t>& indices)  const;

    // 打平
    void flatten();

    // padding操作
    void padding(const std::vector<uint32_t>& pads, T padding_value);

    // 判空操作
    bool empty() const;

    //转换操作
    void transform(std::function<void(T & )> func);

    // 切片操作
    Tensor slice(uint32_t start, uint32_t end) const;
    Tensor slice(uint32_t index) const;

    // 获取矩阵信息
    uint32_t rows() const;
    uint32_t cols() const;
    uint32_t channels() const;


private:
    std::unique_ptr<T[]> data_;        // 存储数据
    std::vector<uint32_t> shape_;        // 存储形状
    std::vector<uint32_t> strides_;      // 存储步长
    // 加载数据
    void initializeData(const std::vector<T>& data);

    // 计算总数
    uint32_t calculateTotalSize(const std::vector<uint32_t>& shape) const;

    // 计算总步长
    void calculateStrides();

    // 元素操作
    Tensor elementWiseOperation(const Tensor& other, std::function<T(T, T)> op) const;

    // 打印Tensor
    void printTensor(uint32_t dimIndex, const std::vector<uint32_t>& indices = {}) const;

    // 计算偏移量
    uint32_t indexToOffset(const std::vector<uint32_t>& indices) const;



};

#endif //INFERNETO_TENSOR_H

}
