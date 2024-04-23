//
// Created by hanke on 2024/4/23.
//

#ifndef INFERNETO_CONVOLUTION_HPP
#define INFERNETO_CONVOLUTION_HPP
#include "node/abstract/param_node.hpp"
namespace infer_neto {
class ConvolutionLayer : public ParamLayer {
public:
    explicit ConvolutionLayer(uint32_t output_channel, uint32_t in_channel,
                              uint32_t kernel_h, uint32_t kernel_w,
                              uint32_t padding_h, uint32_t padding_w,
                              uint32_t stride_h, uint32_t stride_w,
                              uint32_t groups, bool use_bias)
            : ParamLayer("Convolution"),
              use_bias_(use_bias),
              groups_(groups),
              padding_h_(padding_h),
              padding_w_(padding_w),
              stride_h_(stride_h),
              stride_w_(stride_w) {
        if (groups != 1) {
            in_channel /= groups;
        }
        this->InitWeightParam(output_channel, in_channel, kernel_h, kernel_w);
        if (use_bias_) {
            this->InitBiasParam(output_channel, 1, 1, 1);
        }
    };

    InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                        std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;
    /**
     * 初始化kernel的im2col排布
     */
    void InitIm2ColWeight();

    static ParseParameterAttrStatus GetInstance(
            const std::shared_ptr<RuntimeOperator>& op,
            std::shared_ptr<Layer>& conv_layer);
private:
    void ConvGemmBias(const Tensor<float>& input_matrix,
                      const std::shared_ptr<Tensor<float>>& output_tensor,
                      uint32_t group, uint32_t kernel_index,
                      uint32_t kernel_count_group, const Tensor<float>& kernel,
                      uint32_t output_w, uint32_t output_h) const;

    Tensor<float> Im2Col(sftensor input, uint32_t kernel_w, uint32_t kernel_h,
                      uint32_t input_w, uint32_t input_h, uint32_t input_c_group,
                      uint32_t group, uint32_t row_len, uint32_t col_len) const;

    bool use_bias_ = false;
    uint32_t groups_ = 1;
    uint32_t padding_h_ = 0;
    uint32_t padding_w_ = 0;
    uint32_t stride_h_ = 1;
    uint32_t stride_w_ = 1;
    std::vector<Tensor<float>> kernel_matrix_arr_;


};
}
#endif //INFERNETO_CONVOLUTION_HPP
