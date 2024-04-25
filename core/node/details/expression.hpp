//
// Created by hanke on 2024/4/25.
//

#ifndef INFERNETO_EXPRESSION_HPP
#define INFERNETO_EXPRESSION_HPP
#include <utility>

#include "node/abstract/non_param_node.hpp"
#include "node/parser/parse_expression.hpp"
namespace infer_neto {
class ExpressionLayer : public NonParamLayer {
public:
    explicit ExpressionLayer(std::string statement)
    : NonParamLayer("Expression"), statement_(std::move(statement)) {
        parser_ = std::make_unique<ExpressionParser>(statement_);
    }
    InferStatus Forward(
            const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
            std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

    static ParseParameterAttrStatus GetInstance(
            const std::shared_ptr<RuntimeOperator>& op,
            std::shared_ptr<Layer>& expression_layer);
private:
    std::string statement_;
    std::unique_ptr<ExpressionParser> parser_;
};

}
#endif //INFERNETO_EXPRESSION_HPP
