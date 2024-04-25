//
// Created by hanke on 2024/4/25.
//

#ifndef INFERNETO_PARSE_EXPRESSION_HPP
#define INFERNETO_PARSE_EXPRESSION_HPP
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace infer_neto {
// 表达式词的类型
enum class TokenType {
    TokenUnknown = -9,
    TokenInputNumber = -8,
    TokenComma = -7,
    TokenAdd = -6,
    TokenMul = -5,
    TokenLeftBracket = -4,
    TokenRightBracket = -3,
    TokenSin = -2
};
// 表达式词对象
struct Token {
    TokenType token_type = TokenType::TokenUnknown;
    int32_t start_pos = 0;  // 解析词开始位置
    int32_t end_pos = 0;  // 解析词结束位置
    Token(TokenType token_type, int32_t start_pos, int32_t end_pos)
            : token_type(token_type), start_pos(start_pos), end_pos(end_pos) {}
};

// 语法树节点
    struct TokenNode {
        int32_t num_index = -1;
        std::shared_ptr<TokenNode> left = nullptr;   // 语法树的左节点
        std::shared_ptr<TokenNode> right = nullptr;  // 语法树的右节点
        TokenNode(int32_t num_index, std::shared_ptr<TokenNode> left,
                  std::shared_ptr<TokenNode> right);
        TokenNode() = default;
    };


// 表达式解析
// add(add(add(@0,@1),@1),add(@0,@2))
class ExpressionParser {
public:
    explicit ExpressionParser(std::string statement)
            : statement_(std::move(statement)) {}

    /**
     * 词法分析
     * @param retokenize 是否需要重新进行语法分析
     */
    void Tokenizer(bool retokenize = false);

    /**
     * 语法分析
     * @return 生成的语法树
     */
    std::vector<std::shared_ptr<TokenNode>> Generate();

    std::shared_ptr<TokenNode> Generate_(int32_t& index);

    /**
     * 返回词法分析的结果
     * @return 词法分析的结果
     */
    const std::vector<Token>& tokens() const;

    /**
     * 返回词语字符串
     * @return 词语字符串
     */
    const std::vector<std::string>& token_strs() const;

private:
    // 被分割的词语数组
    std::vector<Token> tokens_;
    // 被分割的字符串数组
    std::vector<std::string> token_strs_;
    // 待分割的表达式
    std::string statement_;
};

}


#endif //INFERNETO_PARSE_EXPRESSION_HPP
