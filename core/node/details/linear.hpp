//
// Created by hanke on 2024/4/26.
//

#ifndef INFERNETO_LINEAR_HPP
#define INFERNETO_LINEAR_HPP

#include "node/abstract/param_node.hpp"

namespace infer_neto{
class LinearLayer : public ParamLayer {
public:
LinearLayer(int32_t in_features, int32_t out_features, bool use_bias)
: ParamLayer("Linear"),
    use_bias_(use_bias),
    in_features_(in_features),
    out_features_(out_features) {
        CHECK_GT(in_features_, 0);
        CHECK_GT(out_features_, 0);
        this->InitWeightParam(1, 1, out_features, in_features);
        if (use_bias) {
            this->InitBiasParam(1, 1, 1, out_features);
        }
};

InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                    std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

static ParseParameterAttrStatus GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                            std::shared_ptr<Layer> &linear_layer);

private:
    int32_t in_features_ = 0;
    int32_t out_features_ = 0;
    bool use_bias_ = false;
};
}


#endif //INFERNETO_LINEAR_HPP
