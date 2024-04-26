//
// Created by hanke on 2024/4/26.
//

#ifndef INFERNETO_SOFTMAX_HPP
#define INFERNETO_SOFTMAX_HPP

#include "node/abstract/non_param_node.hpp"

namespace infer_neto {
class SoftmaxLayer : public NonParamLayer {
public:
    explicit SoftmaxLayer(int dim = -1)
    : NonParamLayer("Softmax"), softmax_dim_(dim) {};

    InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                        std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

    static ParseParameterAttrStatus CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                                   std::shared_ptr<Layer>& softmax_layer);

private:
    int softmax_dim_ = -1;
};
}

#endif //INFERNETO_SOFTMAX_HPP
