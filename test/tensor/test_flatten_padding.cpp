//
// Created by fss on 23-6-4.
//
#include "data/cpu/tensor.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

TEST(test_homework, homework1_flatten1) {
    using namespace infer_neto;
    Tensor<float> f1(2, 3, 4);
    f1.Show();
    f1.Flatten();
    f1.Show();
    ASSERT_EQ(f1.shapes().size(), 1);
    ASSERT_EQ(f1.shapes().at(0), 24);
}

TEST(test_homework, homework1_flatten2) {
  using namespace infer_neto;
  Tensor<float> f1(12, 24);
  f1.Flatten();
  ASSERT_EQ(f1.shapes().size(), 1);
  ASSERT_EQ(f1.shapes().at(0), 24 * 12);
}

TEST(test_homework, homework2_padding1) {
  using namespace infer_neto;
  Tensor<float> tensor(3, 4, 5);
  ASSERT_EQ(tensor.channels(), 3);
  ASSERT_EQ(tensor.rows(), 4);
  ASSERT_EQ(tensor.cols(), 5);
  tensor.Fill(1);
  tensor.Padding({1, 2, 3, 4}, 0);
  ASSERT_EQ(tensor.rows(), 7);
  ASSERT_EQ(tensor.cols(), 12);
  tensor.Show();
  int index = 0;
  for (uint32_t c = 0; c < tensor.channels(); ++c) {
    for (uint32_t r = 0; r < tensor.rows(); ++r) {
      for (uint32_t c_ = 0; c_ < tensor.cols(); ++c_) {
        if ((r >= 2 && r <= 4) && (c_ >= 3 && c_ <= 7)) {
          ASSERT_EQ(tensor.at(c, r, c_), 1.f) << c << " "
                                              << " " << r << " " << c_;
        }
        index += 1;
      }
    }
  }
}

TEST(test_homework, homework2_padding2) {
  using namespace infer_neto;
  Tensor<float> tensor(3, 4, 5);
  ASSERT_EQ(tensor.channels(), 3);
  ASSERT_EQ(tensor.rows(), 4);
  ASSERT_EQ(tensor.cols(), 5);

  tensor.Fill(1.f);
  tensor.Padding({2, 2, 2, 2}, 3.14f);
  ASSERT_EQ(tensor.rows(), 8);
  ASSERT_EQ(tensor.cols(), 9);
  int index = 0;
  for (uint32_t c = 0; c < tensor.channels(); ++c) {
    for (uint32_t r = 0; r < tensor.rows(); ++r) {
      for (uint32_t c_ = 0; c_ < tensor.cols(); ++c_) {
        if (c_ <= 1 || r <= 1) {
          ASSERT_EQ(tensor.at(c, r, c_), 3.14f);
        } else if (c >= tensor.cols() - 1 || r >= tensor.rows() - 1) {
          ASSERT_EQ(tensor.at(c, r, c_), 3.14f);
        }
        if ((r >= 2 && r <= 5) && (c_ >= 2 && c_ <= 6)) {
          ASSERT_EQ(tensor.at(c, r, c_), 1.f);
        }
        index += 1;
      }
    }
  }
}