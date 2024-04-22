//
// Created by hanke on 2024/4/21.
//

#ifndef INFERNETO_MAXPOOLING_HPP
#define INFERNETO_MAXPOOLING_HPP
#include "node/abstract/non_param_node.hpp"
namespace infer_neto {
class MaxPoolingLayer : public NonParamLayer {
public:
    MaxPoolingLayer(uint32_t padding_h,
                    uint32_t padding_w,
                    uint32_t pooling_size_h,
                    uint32_t pooling_size_w,
                    uint32_t stride_h,
                    uint32_t stride_w)
                    : NonParamLayer("MaxPooling"),
                    padding_h_(padding_h),
                    padding_w_(padding_w),
                    pooling_size_h_(pooling_size_h),
                    pooling_size_w_(pooling_size_w),
                    stride_h_(stride_h),
                    stride_w_(stride_w){}

    InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>> & inputs,
                        std::vector<std::shared_ptr<Tensor<float>>>& outputs)
                        override;

    static ParseParameterAttrStatus GetInstance(const std::shared_ptr<RuntimeOperator>& op,
                                                std::shared_ptr<Layer>& maxpooling_layer);


private:
    uint32_t padding_h_ = 0;
    uint32_t padding_w_ = 0;
    uint32_t pooling_size_h_ = 0;
    uint32_t pooling_size_w_ = 0;
    uint32_t stride_h_ = 1;
    uint32_t stride_w_ = 1;
};
}
#endif //INFERNETO_MAXPOOLING_HPP
