//
// Created by hanke on 2024/3/31.
//
#include <glog/logging.h>
#include <random>
#include <functional>
#include "tensor.hpp"
namespace infer_neto {
    Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
        size_ = channels * rows * cols;
        data_ = std::make_unique<float[]>(size_);
        if (channels == 1 && rows == 1) {
            this->raw_shapes_ = std::vector<uint32_t>{cols};
        } else if (channels == 1) {
            this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
        } else {
            this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
        }
        calculateStrides();
    }

    Tensor<float>::Tensor(uint32_t size) {
        size_ = size;
        data_ = std::make_unique<float[]>(size_);
        this->raw_shapes_ = std::vector<uint32_t>{size};
        calculateStrides();
    }

    Tensor<float>::Tensor(uint32_t rows, uint32_t cols) {
        size_ = rows * cols;
        data_ = std::make_unique<float[]>(rows * cols * 1);
        this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
        calculateStrides();
    }

    Tensor<float>::Tensor(const std::vector<uint32_t> &shapes) {
        CHECK(!shapes.empty() && shapes.size() <= 3);

        uint32_t remaining = 3 - shapes.size();
        std::vector<uint32_t> shapes_(3, 1);
        std::copy(shapes.begin(), shapes.end(), shapes_.begin() + remaining);

        uint32_t channels = shapes_.at(0);
        uint32_t rows = shapes_.at(1);
        uint32_t cols = shapes_.at(2);
        size_ = rows * cols * channels;
        data_ = std::make_unique<float[]>(size_);
        if (channels == 1 && rows == 1) {
            this->raw_shapes_ = std::vector<uint32_t>{cols};
        } else if (channels == 1) {
            this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
        } else {
            this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
        }
        calculateStrides();
    }

    Tensor<float>::Tensor(const Tensor &tensor) {
        // 复制形状和步长信息
        this->raw_shapes_ = tensor.raw_shapes_;
        this->strides_ = tensor.strides_;
        this->size_ = tensor.size_;

        // 为data_分配空间并进行深拷贝
        this->data_ = std::make_unique<float[]>(size_);
        std::copy(tensor.data_.get(), tensor.data_.get() + size_, this->data_.get());
    }

    Tensor<float>::Tensor(Tensor<float> &&tensor) noexcept {
        if (this != &tensor) {
            this->data_ = std::move(tensor.data_);
            this->raw_shapes_ = std::move(tensor.raw_shapes_);
            this->strides_ = std::move(tensor.strides_);
            this->size_ = tensor.size_;
        }
    }

    Tensor<float> &Tensor<float>::operator=(Tensor<float> &&tensor) noexcept {
        if (this != &tensor) {
            this->data_ = std::move(tensor.data_);
            this->raw_shapes_ = std::move(tensor.raw_shapes_);
            this->strides_ = std::move(tensor.strides_);
            this->size_ = tensor.size_;
        }
        return *this;
    }

    Tensor<float> &Tensor<float>::operator=(const Tensor &tensor) {
        if (this != &tensor) {
            this->strides_ = tensor.strides_;
            this->raw_shapes_ = tensor.raw_shapes_;
            this->size_ = tensor.size_;

            // 为data_分配空间并进行深拷贝
            this->data_ = std::make_unique<float[]>(size_);
            std::copy(tensor.data_.get(), tensor.data_.get() + size_, this->data_.get());
        }
        return *this;
    }


    void Tensor<float>::calculateStrides() {
        strides_.resize(raw_shapes_.size());
        uint32_t stride = 1;
        for (size_t i = raw_shapes_.size(); i-- > 0;) {
            strides_[i] = stride;
            stride *= raw_shapes_[i];
        }
    }


    uint32_t Tensor<float>::rows() const {
        if (this->raw_shapes_.size() >= 2) {
            // 对于至少有二维的张量，倒数第二个元素是行数
            return this->raw_shapes_[this->raw_shapes_.size() - 2];
        } else {
            // 对于一维张量，我们可以认为行数为1
            return 1;
        }
    }

    uint32_t Tensor<float>::cols() const {
        if (!this->raw_shapes_.empty()) {
            // 最后一个元素总是列数
            return this->raw_shapes_.back();
        } else {
            // 如果张量是空的，列数为0
            return 0;
        }
    }

    uint32_t Tensor<float>::channels() const {
        if (this->raw_shapes_.size() == 3) {
            // 对于至少有三维的张量，第一个元素是通道数
            return this->raw_shapes_[0];
        } else {
            // 对于二维或一维张量，我们可以认为通道数为1
            return 1;
        }
    }


    void Tensor<float>::set_data(const float *&data) {
        CHECK(!data);
        // 为data_分配内存。这里我们假设每次调用set_data都会重置数据，
        // 如果想要优化内存使用，可以根据需要进行调整
        data_ = std::make_unique<float[]>(size_);

        // 使用std::copy复制数据。注意，我们这里没有直接的方式知道数据的实际大小，
        // 所以我们依赖于调用者正确地维护数据大小与Tensor形状的一致性
        std::copy(data, data + size_, data_.get());
    }

    bool Tensor<float>::empty() const { return !this->data_; }

    float Tensor<float>::index(uint32_t offset) const {
        CHECK(offset < size_) << "Tensor index out of bound!";
        return this->data_[offset];
    }

    float &Tensor<float>::index(uint32_t offset) {
        CHECK(offset < size_) << "Tensor index out of bound!";
        return this->data_[offset];
    }

    std::vector<uint32_t> Tensor<float>::shapes() const {
        CHECK(this->data_);
        return {this->channels(), this->rows(), this->cols()};
    }

    std::unique_ptr<float[]> &Tensor<float>::data() { return this->data_; }

    const std::unique_ptr<float[]> &Tensor<float>::data() const { return this->data_; }

    float *Tensor<float>::slice(uint32_t channel) {
        CHECK_LE (channel, raw_shapes_[0]) << "Channel index out of range.";
        return this->data_.get() + channel * strides_[0];
    }

    float *Tensor<float>::slice(uint32_t channel) const {
        CHECK_LE (channel, raw_shapes_[0]) << "Channel index out of range.";
        return this->data_.get() + channel * strides_[0];
    }

    float Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) const {
        CHECK_LT(row, this->rows());
        CHECK_LT(col, this->cols());
        CHECK_LT(channel, this->channels());
        return this->data_[(channel * this->rows() * this->cols()) + (row * this->cols()) + col];
    }

    float &Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) {
        CHECK_LT(row, this->rows());
        CHECK_LT(col, this->cols());
        CHECK_LT(channel, this->channels());
        return this->data_[(channel * this->rows() * this->cols()) + (row * this->cols()) + col];
    }

    void Tensor<float>::Padding(const std::vector<uint32_t> &pads, float padding_value) {
        CHECK(this->data_) << "The data area of the tensor is empty.";
        CHECK_EQ(pads.size(), 4) << "Padding dimensions must be 4.";

        uint32_t pad_rows1 = pads.at(0);  // 上方填充行数
        uint32_t pad_rows2 = pads.at(1);  // 下方填充行数
        uint32_t pad_cols1 = pads.at(2);  // 左侧填充列数
        uint32_t pad_cols2 = pads.at(3);  // 右侧填充列数

        uint32_t new_rows = this->rows() + pad_rows1 + pad_rows2;
        uint32_t new_cols = this->cols() + pad_cols1 + pad_cols2;
        uint32_t new_channels = this->channels();

        // 创建新的数据数组
        std::unique_ptr<float[]> new_data = std::make_unique<float[]>(new_channels * new_rows * new_cols);
        std::fill_n(new_data.get(), new_channels * new_rows * new_cols, padding_value);

        // 复制原始数据到新的数据数组中正确的位置
        for (uint32_t ch = 0; ch < new_channels; ++ch) {
            for (uint32_t r = 0; r < this->rows(); ++r) {
                std::copy(this->data_.get() + (ch * this->rows() + r) * this->cols(),
                          this->data_.get() + (ch * this->rows() + r + 1) * this->cols(),
                          new_data.get() + ((ch * new_rows + r + pad_rows1) * new_cols + pad_cols1));
            }
        }

        // 更新类成员变量
        this->data_ = std::move(new_data);
        this->raw_shapes_ = std::vector<uint32_t>{new_channels, new_rows, new_cols};
        this->calculateStrides();  // 更新步长信息
    }


    void Tensor<float>::Fill(float value) {
        CHECK(this->data_);
        std::fill_n(this->data_.get(), this->size_, value);
    }

    void Tensor<float>::Fill(const std::vector<float> &values) {
        CHECK(this->data_);
        const uint32_t total_elems = this->size_;
        CHECK_EQ(values.size(), total_elems);

        std::copy(values.begin(), values.end(), this->data_.get());

    }

    void Tensor<float>::Show() {
        if (this->empty()) {
            LOG(INFO) << "Tensor is empty!";
            return;
        }

        // 迭代每一个通道
        for (uint32_t c = 0; c < this->channels(); ++c) {
            LOG(INFO) << "Channel " << c << ":";

            // 获取当前通道的数据指针
            float* channel_data = this->slice(c);
            if (!channel_data) {
                LOG(INFO) << "  Null data for channel " << c;
                continue;
            }

            // 使用一个字符串流来收集一行的数据
            std::ostringstream stream;
            for (uint32_t r = 0; r < this->rows(); ++r) {
                for (uint32_t col = 0; col < this->cols(); ++col) {
                    // 根据行主顺序计算索引
                    uint32_t index = r * this->cols() + col;
                    stream << channel_data[index] << " ";
                }
                LOG(INFO) << stream.str();
                stream.str(""); // 清空流以便下一行使用
                stream.clear(); // 清除任何错误状态
            }
        }
    }

    void Tensor<float>::Flatten() {
        CHECK(this->data_);
        this->raw_shapes_ = {this->size()};
        this->calculateStrides();
    }

    void Tensor<float>::Rand() {
        CHECK(this->data_);
        // 创建随机数发生器
        std::random_device rd;  // 非确定性随机数生成器
        std::mt19937 gen(rd());  // 以 rd() 为种子，初始化 Mersenne Twister 伪随机数生成器
        std::uniform_real_distribution<> dis(0.0, 1.0);  // 定义均匀分布范围

        // 为每个张量元素生成随机数
        for (std::uint32_t i = 0; i < this->size_; ++i) {
            this->data_[i] = dis(gen);
        }
    }

    void Tensor<float>::Ones() {
        CHECK(this->data_);
        this->Fill(1.f);
    }

    void Tensor<float>::Transform(const std::function<float(float)> &filter) {
        CHECK(this->data_);
        for (uint32_t i = 0; i < this->size_; ++i) {
            data_[i] = filter(data_[i]);
        }
    }

    const std::vector<uint32_t> &Tensor<float>::raw_shapes() const {
        CHECK(!this->raw_shapes_.empty());
        CHECK_LE(this->raw_shapes_.size(), 3);
        CHECK_GE(this->raw_shapes_.size(), 1);
        return this->raw_shapes_;
    }

    void Tensor<float>::Reshape(const std::vector<uint32_t> &shapes) {
        CHECK(this->data_);
        CHECK(!shapes.empty());
        const uint32_t origin_size = this->size_;
        const uint32_t current_size = std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies());
        CHECK(shapes.size() <= 3);
        CHECK(current_size == origin_size);

        this->raw_shapes_ = shapes;
        this->calculateStrides();
    }

    float *Tensor<float>::raw_ptr() {
        CHECK(this->data_);
        return this->data_.get();;
    }

    float *Tensor<float>::raw_ptr(uint32_t offset) {
        const uint32_t size = this->size_;
        CHECK(this->data_);
        CHECK_LT(offset, size);
        return this->data_.get() + offset;
    }

    std::vector<float> Tensor<float>::values() {
        CHECK(this->data_);
        std::vector<float> values(this->size());

        std::copy(this->data_.get(), this->data_.get() + this->size(),
                  values.begin());

        return values;
    }

    float *Tensor<float>::matrix_raw_ptr(uint32_t index) {
        CHECK_LT(index, this->channels());
        uint32_t offset = index * this->rows() * this->cols();
        CHECK_LE(offset, this->size());
        float *mem_ptr = this->raw_ptr() + offset;
        return mem_ptr;
    }

    uint32_t Tensor<float>::size() const {
        return this->size_;
    }

    sftensor operator-=(sftensor tensor, const float value) {
        CHECK(tensor != nullptr);
        float* data = tensor->raw_ptr();
        for(int i = 0; i < tensor->size(); i++) {
            data[i] -= value;
        }
        return tensor;
    }

}