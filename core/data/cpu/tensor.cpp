//
// Created by hanke on 2024/3/31.
//
#include <glog/logging.h>
#include "tensor.hpp"
namespace infer_neto {

template<typename T>
Tensor<T>::Tensor() : data_(nullptr), shape_({}), strides_({}) {}

template<typename T>
Tensor<T>::Tensor(const std::vector<uint32_t>& shape, const std::vector<T>& data) : shape_(shape) {
    initializeData(data);
}

template<typename T>
Tensor<T>::Tensor(const std::vector<uint32_t>& shape, T fillValue) : shape_(shape) {
    initializeData(std::vector<T>(calculateTotalSize(shape), fillValue));
}

template<typename T>
Tensor<T>::Tensor(const Tensor<T>& other) {
    // 实现复制构造函数的细节
    shape_ = other.shape_;
    strides_ = other.strides_;
    auto totalSize = calculateTotalSize(shape_);
    data_ = std::make_unique<T[]>(totalSize);
    std::copy(other.data_.get(), other.data_.get() + totalSize, data_.get());
}

template<typename T>
Tensor<T>::Tensor(Tensor<T>&& other) noexcept
        : data_(std::move(other.data_)), shape_(std::move(other.shape_)), strides_(std::move(other.strides_)) {
    // 移动构造函数体可以保持空，因为成员初始化列表已经完成了所有工作
}

template<typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>& other) {
    if (this != &other) {
        shape_ = other.shape_;
        strides_ = other.strides_;
        auto totalSize = calculateTotalSize(shape_);
        data_ = std::make_unique<T[]>(totalSize);
        std::copy(other.data_.get(), other.data_.get() + totalSize, data_.get());
    }
    return *this;
}

template<typename T>
Tensor<T>& Tensor<T>::operator=(Tensor<T>&& other) noexcept {
    if (this != &other) {
        data_ = std::move(other.data_);
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
    }
    return *this;
}


template<typename T>
Tensor<T> Tensor<T>::operator+(const Tensor& other) const {
    return elementWiseOperation(other, std::plus<>());
}

template<typename T>
Tensor<T> Tensor<T>::operator-(const Tensor& other) const {
    return elementWiseOperation(other, std::minus<>());
}

template<typename T>
Tensor<T> Tensor<T>::operator*(const Tensor& other) const {
    return elementWiseOperation(other, std::multiplies<>());
}

template<typename T>
Tensor<T> Tensor<T>::operator/(const Tensor& other) const {
    return elementWiseOperation(other, std::divides<>());
}

template<typename T>
void Tensor<T>::random(T lower_bound, T upper_bound) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(lower_bound, upper_bound);

    for (uint32_t i = 0; i < calculateTotalSize(shape_); ++i) {
        data_[i] = static_cast<T>(dis(gen));
    }
}

template<typename T>
void Tensor<T>::fill(T data) {
    for (uint32_t i = 0; i < calculateTotalSize(shape_); ++i) {
        data_[i] = data;
    }
}

template<typename T>
void Tensor<T>::fill(const std::vector<T>& values) {
    uint32_t total_elems = calculateTotalSize(shape_);
    CHECK_EQ(values.size(), total_elems) << "Values size does not match tensor's total elements.";
    // 对于行优先，我们直接复制values到data_，因为标准的C++数组（std::unique_ptr<T[]>指向的数组）本身就是行优先存储
    std::copy(values.begin(), values.end(), data_.get());
}
template<typename T>
Tensor<T> Tensor<T>::transpose() const {
    CHECK_EQ(shape_.size(), 2) << "Transpose is only implemented for 2D tensors.";

    std::vector<uint32_t> transposedShape = {shape_[1], shape_[0]};
    Tensor transposedTensor(transposedShape);

    for (uint32_t i = 0; i < shape_[0]; ++i) {
        for (uint32_t j = 0; j < shape_[1]; ++j) {
            transposedTensor.data_[j * shape_[0] + i] = data_[i * shape_[1] + j];
        }
    }

    return transposedTensor;
}

template<typename T>
void Tensor<T>::reshape(const std::vector<uint32_t>& newShape) {
    uint32_t currentTotalSize = calculateTotalSize(shape_);
    uint32_t newTotalSize = calculateTotalSize(newShape);
    CHECK_EQ(currentTotalSize, newTotalSize) << "Total size of new shape must be unchanged.";

    // 更新形状
    shape_ = newShape;

    // 如果你的Tensor类使用步长来计算索引，则需要重新计算步长
    calculateStrides();
}

template<typename T>
void Tensor<T>::print() const {
    printTensor(0); // 直接调用，依赖于方法签名中的默认参数
    std::cout << std::endl;
}


template<typename T>
T& Tensor<T>::at(const std::vector<uint32_t>& indices) {
    return data_[indexToOffset(indices)];
}


template<typename T>
const T& Tensor<T>::at(const std::vector<uint32_t>& indices) const {
    return data_[indexToOffset(indices)];
}

template<typename T>
bool Tensor<T>::empty() const {
    return !data_ || calculateTotalSize(shape_) == 0;
}

template<typename T>
void Tensor<T>::transform(std::function<void(T&)> func) {
    for (size_t i = 0; i < calculateTotalSize(shape_); ++i) {
        func(data_[i]);
    }
}

template<typename T>
void Tensor<T>::flatten() {
    // 计算总元素数
    uint32_t totalSize = calculateTotalSize(shape_);

    // 更新形状为一维，其中包含所有元素
    shape_ = {totalSize};

    // 由于现在是一维张量，步长只需要是1
    strides_ = {1}; // 假设你使用步长来计算索引
}

template<typename T>
Tensor<T> Tensor<T>::slice(uint32_t start, uint32_t end) const {
    CHECK_LT(start, end) << "start should be less than or equal to end.";
    CHECK_LT(end, shape_[0]) << "end should be less than shape_[0].";
    std::vector<uint32_t> newShape = shape_;
    newShape[0] = end - start;
    std::vector<T> newData(newShape[0] * calculateTotalSize(std::vector<uint32_t>(shape_.begin() + 1, shape_.end())));

    for (uint32_t i = start; i < end; ++i) {
        std::copy(data_.get() + i * calculateTotalSize(std::vector<uint32_t>(shape_.begin() + 1, shape_.end())),
                  data_.get() + (i + 1) * calculateTotalSize(std::vector<uint32_t>(shape_.begin() + 1, shape_.end())),
                  newData.begin() + (i - start) * calculateTotalSize(std::vector<uint32_t>(shape_.begin() + 1, shape_.end())));
    }

    return Tensor(newShape, newData);
}
template<typename T>
Tensor<T> Tensor<T>::slice(uint32_t index) const {
    CHECK_LE(index, shape_[0]) << "Index should be less than the first dimension of the shape.";

    // 确保张量至少是二维的，因为切片会减少一个维度
    CHECK_GE(shape_.size(), 2) << "Tensor must be at least 2-dimensional for slicing.";

    // 创建新形状，移除最外层维度
    std::vector<uint32_t> newShape(shape_.begin() + 1, shape_.end());

    // 计算新形状的总大小
    uint32_t newSize = calculateTotalSize(newShape);

    // 为新张量的数据分配空间
    std::vector<T> newData(newSize);

    // 计算起始偏移量
    uint32_t startOffset = index * newSize;

    // 复制数据到新张量
    std::copy(data_.get() + startOffset, data_.get() + startOffset + newSize, newData.begin());

    // 创建并返回新的张量
    return Tensor<T>(newShape, newData);
}

template<typename T>
void Tensor<T>::padding(const std::vector<uint32_t>& pads, T padding_value) {
    // 确保pads的大小是4（上下左右）
    CHECK_EQ(pads.size(), 4) << "Padding requires 4 values for top, bottom, left, and right pads.";

    // 获取原始维度
    uint32_t original_rows = rows();
    uint32_t original_cols = cols();
    uint32_t original_channels = channels();

    // 计算新维度
    uint32_t new_rows = original_rows + pads[0] + pads[1];
    uint32_t new_cols = original_cols + pads[2] + pads[3];

    // 创建新形状
    std::vector<uint32_t> new_shape = {original_channels, new_rows, new_cols};

    // 分配新数据
    std::unique_ptr<T[]> new_data(new T[calculateTotalSize(new_shape)]);
    std::fill_n(new_data.get(), calculateTotalSize(new_shape), padding_value);

    // 复制原始数据到新数据中，考虑填充
    for (uint32_t c = 0; c < original_channels; ++c) {
        for (uint32_t r = 0; r < original_rows; ++r) {
            for (uint32_t col = 0; col < original_cols; ++col) {
                uint32_t old_index = c * original_rows * original_cols + r * original_cols + col;
                uint32_t new_index = c * new_rows * new_cols + (r + pads[0]) * new_cols + (col + pads[2]);
                new_data[new_index] = data_[old_index];
            }
        }
    }

    // 更新Tensor状态
    data_ = std::move(new_data);
    shape_ = new_shape;
    calculateStrides(); // 重新计算步长，如果你的Tensor类使用步长
}

template<typename T>
uint32_t Tensor<T>::rows() const {
    // 检查至少是一个二维张量
    if (shape_.size() >= 2) {
        return static_cast<uint32_t>(shape_[1]);
    } else {
        return static_cast<uint32_t>(1);
    }
}

template<typename T>
uint32_t Tensor<T>::cols() const {
    // 检查至少是一个1维张量
    if (!shape_.empty()) {
        return static_cast<uint32_t>(shape_[2]);
    } else {
        return static_cast<uint32_t>(1);
    }
}

template<typename T>
uint32_t Tensor<T>::channels() const {
    // 检查至少是一个三维张量
    if (shape_.size() == 3) {
        return static_cast<uint32_t>(shape_[0]);
    } else {
        return static_cast<uint32_t>(1);
    }
}

template<typename T>
void Tensor<T>::show() {
    print();
}

template<typename T>
const std::vector<uint32_t>& Tensor<T>::shape() const {
    return shape_;
}

template<typename T>
uint32_t Tensor<T>::size() const {
    return calculateTotalSize(shape_);
}

template<typename T>
uint32_t Tensor<T>::calculateTotalSize(const std::vector<uint32_t>& shape) const {
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint32_t>());
}

template<typename T>
void Tensor<T>::initializeData(const std::vector<T>& data) {
    uint32_t totalSize = calculateTotalSize(shape_);
    CHECK_EQ(totalSize, data.size()) << "Data size does not match tensor shape.";
    data_ = std::make_unique<T[]>(totalSize);
    std::copy(data.begin(), data.end(), data_.get());
    calculateStrides();
}

template<typename T>
void Tensor<T>::calculateStrides() {
    strides_.resize(shape_.size());
    uint32_t stride = 1;
    for (int i = shape_.size() - 1; i >= 0; --i) {
        strides_[i] = stride;
        stride *= shape_[i];
    }
}

template<typename T>
Tensor<T> Tensor<T>::elementWiseOperation(const Tensor& other, std::function<T(T, T)> op) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes do not match.");
    }
    std::vector<T> resultData(calculateTotalSize(shape_));
    for (uint32_t i = 0; i < resultData.size(); i++) {
        resultData[i] = op(data_[i], other.data_[i]);
    }
    return Tensor(shape_, resultData);
}

template<typename T>
void Tensor<T>::printTensor(uint32_t dimIndex, const std::vector<uint32_t>& indices) const {
    std::stringstream ss; // 使用stringstream来构建字符串

    if (dimIndex >= shape_.size()) {
        // 当索引完全时，打印该元素
        ss << data_[indexToOffset(indices)];
    } else {
        ss << "[";
        for (uint32_t i = 0; i < shape_[dimIndex]; ++i) {
            // 构建当前维度的索引
            std::vector<uint32_t> currentIndices = indices;
            currentIndices.push_back(i);
            // 递归打印下一个维度，但不直接打印，而是追加到stringstream
            std::stringstream temp_ss;
            std::streambuf* orig_buf = std::cout.rdbuf(temp_ss.rdbuf());
            printTensor(dimIndex + 1, currentIndices);
            std::cout.rdbuf(orig_buf); // 恢复原来的streambuf

            ss << temp_ss.str();
            if (i < shape_[dimIndex] - 1) {
                if(dimIndex == shape_.size() - 2) ss << "\n"; // 最内层维度结束后换行
                else ss << ", ";
            }
        }
        ss << "]";
    }

    if (dimIndex == 0) {
        LOG(INFO) << ss.str(); // 最外层结束后使用glog进行输出
    } else {
        std::cout << ss.str(); // 非最外层仍然使用std::cout进行递归构建
    }
}

template<typename T>
uint32_t Tensor<T>::indexToOffset(const std::vector<uint32_t>& indices) const {
//    CHECK_EQ(indices.size(), shape_.size()) << "Index dimension does not match tensor dimension.";
    uint32_t offset = 0;
    for (uint32_t i = 0; i < indices.size(); ++i) {
        if (indices[i] == 1) continue;
        CHECK_LE(indices[i], shape_[i]) << "Tensor index out of range.";
        offset += indices[i] * strides_[i];
    }
    return offset;
}

template class Tensor<float>;
template class Tensor<int>;
}