//
// Created by hanke on 2024/4/26.
//

#ifndef INFERNETO_ADAPTIVE_AVGPOOLING_HPP
#define INFERNETO_ADAPTIVE_AVGPOOLING_HPP

#include "node/abstract/non_param_node.hpp"
namespace infer_neto {
class AdaptiveAveragePoolingLayer : public NonParamLayer {
public:
    explicit AdaptiveAveragePoolingLayer(uint32_t output_h, uint32_t output_w)
            : NonParamLayer("AdaptiveAveragePooling"), output_h_(output_h), output_w_(output_w) {
        CHECK_GT(output_h_, 0);
        CHECK_GT(output_w_, 0);
    }

    InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                        std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

    static ParseParameterAttrStatus CreateInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                   std::shared_ptr<Layer> &avg_layer);

private:
    uint32_t output_h_ = 0;
    uint32_t output_w_ = 0;
};
}
#endif //INFERNETO_ADAPTIVE_AVGPOOLING_HPP
