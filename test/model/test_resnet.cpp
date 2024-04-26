//
// Created by fss on 23-8-5.
//
#include <gtest/gtest.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "infer/infer_ir.hpp"
#include "node/details/softmax.hpp"


using namespace infer_neto;

infer_neto::sftensor PreProcessImage(const cv::Mat &image) {
    using namespace infer_neto;
    assert(!image.empty());
    // 调整输入大小
    cv::Mat resize_image;
    cv::resize(image, resize_image, cv::Size(224, 224));

    cv::Mat rgb_image;
    cv::cvtColor(resize_image, rgb_image, cv::COLOR_BGR2RGB);

    rgb_image.convertTo(rgb_image, CV_32FC3);
    std::vector<cv::Mat> split_images;
    cv::split(rgb_image, split_images);
    uint32_t input_w = 224;
    uint32_t input_h = 224;
    uint32_t input_c = 3;
    sftensor input = std::make_shared<ftensor>(input_c, input_h, input_w);

    uint32_t index = 0;
    for (const auto &split_image : split_images) {
    assert(split_image.total() == input_w * input_h);
    const cv::Mat &split_image_t = split_image.t();
    memcpy(input->slice(index), split_image_t.data,
           sizeof(float) * split_image.total());
    index += 1;
    }

    float mean_r = 0.485f;
    float mean_g = 0.456f;
    float mean_b = 0.406f;

    float var_r = 0.229f;
    float var_g = 0.224f;
    float var_b = 0.225f;
    assert(input->channels() == 3);
    input->Scale(1.0f / 255.0f); // 将所有像素值缩放到[0, 1]范围
    input->NormalizeChannel(0, mean_r, var_r); // 对第一个通道进行规范化
    input->NormalizeChannel(1, mean_g, var_g); // 对第二个通道进行规范化
    input->NormalizeChannel(2, mean_b, var_b); // 对第三个通道进行规范化
  return input;
}

TEST(test_network, resnet1) {
  using namespace infer_neto;
  const std::string &param_path = "../model_file/resnet18_batch1.pnnx.param";
  const std::string &weight_path = "../model_file/resnet18_batch1.pnnx.bin";
  RuntimeGraph graph(param_path, weight_path);
  graph.Build("pnnx_input_0", "pnnx_output_0");

  const uint32_t batch_size = 1;
  std::vector<sftensor> inputs;
  const std::string &path("../model_file/car.jpg");
  for (uint32_t i = 0; i < batch_size; ++i) {
    cv::Mat image = cv::imread(path);
    // 图像预处理
    sftensor input = PreProcessImage(image);
    inputs.push_back(input);
  }
  auto outputs = graph.Forward(inputs, true);
  ASSERT_EQ(outputs.size(), batch_size);
  outputs[0]->Show();
  SoftmaxLayer softmax_layer(0);
  std::vector<sftensor> outputs_softmax(batch_size);
  softmax_layer.Forward(outputs, outputs_softmax);
  assert(outputs_softmax.size() == batch_size);

  for (int i = 0; i < outputs_softmax.size(); ++i) {
    const sftensor &output_tensor = outputs_softmax.at(i);
    assert(output_tensor->size() == 1 * 1000);
    // 找到类别概率最大的种类
    float max_prob = -1;
    int max_index = -1;
    for (int j = 0; j < output_tensor->size(); ++j) {
      float prob = output_tensor->index(j);
      if (max_prob <= prob) {
        max_prob = prob;
        max_index = j;
      }
    }
    printf("class with max prob is %f index %d\n", max_prob, max_index);
  }
}