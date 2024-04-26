//
// Created by hanke on 2024/4/26.
//

#ifndef INFERNETO_FLATTEN_HPP
#define INFERNETO_FLATTEN_HPP


#include "node/abstract/non_param_node.hpp"

namespace infer_neto{
class FlattenLayer : public NonParamLayer {
public:
    explicit FlattenLayer(int start_dim, int end_dim)
    : NonParamLayer("Flatten"), start_dim_(start_dim), end_dim_(end_dim) {};

    InferStatus Forward(
            const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

    static ParseParameterAttrStatus CreateInstance(
            const std::shared_ptr<RuntimeOperator>& op,
            std::shared_ptr<Layer>& flatten_layer);

private:
    int start_dim_ = 0;
    int end_dim_ = 0;
};
}


#endif //INFERNETO_FLATTEN_HPP
