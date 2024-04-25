//
// Created by fss on 23-7-22.
//
#include "node/abstract/node_factory.hpp"
//#include "node/parser/expression.hpp"

#include "node/parser/parse_expression.hpp"
#include "node/details/expression.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <valarray>

using namespace infer_neto;

TEST(test_parser, tokenizer) {
  using namespace infer_neto;
  const std::string &str = "add(@0,mul(@1,@2))";
  ExpressionParser parser(str);
  parser.Tokenizer();
  const auto &tokens = parser.tokens();
  ASSERT_EQ(tokens.empty(), false);

  const auto &token_strs = parser.token_strs();

  ASSERT_EQ(token_strs.at(0), "add");
  ASSERT_EQ(tokens.at(0).token_type, TokenType::TokenAdd);

  ASSERT_EQ(token_strs.at(1), "(");
  ASSERT_EQ(tokens.at(1).token_type, TokenType::TokenLeftBracket);

  ASSERT_EQ(token_strs.at(2), "@0");
  ASSERT_EQ(tokens.at(2).token_type, TokenType::TokenInputNumber);

  ASSERT_EQ(token_strs.at(3), ",");
  ASSERT_EQ(tokens.at(3).token_type, TokenType::TokenComma);

  ASSERT_EQ(token_strs.at(4), "mul");
  ASSERT_EQ(tokens.at(4).token_type, TokenType::TokenMul);

  ASSERT_EQ(token_strs.at(5), "(");
  ASSERT_EQ(tokens.at(5).token_type, TokenType::TokenLeftBracket);

  ASSERT_EQ(token_strs.at(6), "@1");
  ASSERT_EQ(tokens.at(6).token_type, TokenType::TokenInputNumber);

  ASSERT_EQ(token_strs.at(7), ",");
  ASSERT_EQ(tokens.at(7).token_type, TokenType::TokenComma);

  ASSERT_EQ(token_strs.at(8), "@2");
  ASSERT_EQ(tokens.at(8).token_type, TokenType::TokenInputNumber);

  ASSERT_EQ(token_strs.at(9), ")");
  ASSERT_EQ(tokens.at(9).token_type, TokenType::TokenRightBracket);

  ASSERT_EQ(token_strs.at(10), ")");
  ASSERT_EQ(tokens.at(10).token_type, TokenType::TokenRightBracket);
}

TEST(test_parser, generate1) {
  using namespace infer_neto;
  const std::string &str = "add(@0,@1)";
  ExpressionParser parser(str);
  parser.Tokenizer();
  int index = 0; // 从0位置开始构建语法树
  // 抽象语法树:
  //
  //    add
  //    /  \
  //  @0    @1

  const auto &node = parser.Generate_(index);
  ASSERT_EQ(node->num_index, int(TokenType::TokenAdd));
  ASSERT_EQ(node->left->num_index, 0);
  ASSERT_EQ(node->right->num_index, 1);
}

TEST(test_parser, generate2) {
  using namespace infer_neto;
  const std::string &str = "add(mul(@0,@1),@2)";
  ExpressionParser parser(str);
  parser.Tokenizer();
  int index = 0; // 从0位置开始构建语法树
  // 抽象语法树:
  //
  //       add
  //       /  \
  //     mul   @2
  //    /   \
  //  @0    @1

  const auto &node = parser.Generate_(index);
  ASSERT_EQ(node->num_index, int(TokenType::TokenAdd));
  ASSERT_EQ(node->left->num_index, int(TokenType::TokenMul));
  ASSERT_EQ(node->left->left->num_index, 0);
  ASSERT_EQ(node->left->right->num_index, 1);

  ASSERT_EQ(node->right->num_index, 2);
}
//
TEST(test_parser, reverse_polish) {
  using namespace infer_neto;
  const std::string &str = "add(mul(@0,@1),@2)";
  ExpressionParser parser(str);
  parser.Tokenizer();
  // 抽象语法树:
  //
  //       add
  //       /  \
  //     mul   @2
  //    /   \
  //  @0    @1

  const auto &vec = parser.Generate();
  for (const auto &item : vec) {
    if (item->num_index == -5) {
      LOG(INFO) << "Mul";
    } else if (item->num_index == -6) {
      LOG(INFO) << "Add";
    } else {
      LOG(INFO) << item->num_index;
    }
  }
}

TEST(test_expression, complex1) {
    using namespace infer_neto;
    const std::string expr = "mul(@2,add(@0,@1))";
    ExpressionLayer layer(expr);

    std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(3, 224, 224);
    input1->Fill(2.f);  // Fill tensor with 2.0

    std::shared_ptr<Tensor<float>> input2 = std::make_shared<Tensor<float>>(3, 224, 224);
    input2->Fill(3.f);  // Fill tensor with 3.0

    std::shared_ptr<Tensor<float>> input3 = std::make_shared<Tensor<float>>(3, 224, 224);
    input3->Fill(4.f);  // Fill tensor with 4.0

    std::vector<std::shared_ptr<Tensor<float>>> inputs = {input1, input2, input3};
    std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
    outputs.at(0) = std::make_shared<Tensor<float>>(3, 224, 224);  // Prepare output tensor

    InferStatus status = layer.Forward(inputs, outputs);
    assert(status == InferStatus::kInferSuccess);  // Check that computation was successful

    std::shared_ptr<Tensor<float>> expected_output = std::make_shared<Tensor<float>>(3, 224, 224);
    expected_output->Fill(20.f);  // Expected output: (2+3)*4 = 20 for each element

    std::shared_ptr<Tensor<float>> output1 = outputs.front();
    bool are_equal = true;
    for (uint32_t channel = 0; channel < output1->channels(); ++channel) {
        for (uint32_t row = 0; row < output1->rows(); ++row) {
            for (uint32_t col = 0; col < output1->cols(); ++col) {
                if (std::abs(output1->at(channel, row, col) - expected_output->at(channel, row, col)) > 1e-5) {
                    are_equal = false;
                    break;
                }
            }
            if (!are_equal) break;
        }
        if (!are_equal) break;
    }
    assert(are_equal);
}

TEST(test_parser, tokenizer_sin) {
    using namespace infer_neto;
    const std::string &str = "add(sin(@0),@1)";
    ExpressionParser parser(str);
    parser.Tokenizer();
    const auto &tokens = parser.tokens();
    ASSERT_EQ(tokens.empty(), false);

    const auto &token_strs = parser.token_strs();
    ASSERT_EQ(token_strs.at(0), "add");
    ASSERT_EQ(tokens.at(0).token_type, TokenType::TokenAdd);

    ASSERT_EQ(token_strs.at(1), "(");
    ASSERT_EQ(tokens.at(1).token_type, TokenType::TokenLeftBracket);

    ASSERT_EQ(token_strs.at(2), "sin");
    ASSERT_EQ(tokens.at(2).token_type, TokenType::TokenSin);

    ASSERT_EQ(token_strs.at(3), "(");
    ASSERT_EQ(tokens.at(3).token_type, TokenType::TokenLeftBracket);

    ASSERT_EQ(token_strs.at(4), "@0");
    ASSERT_EQ(tokens.at(4).token_type, TokenType::TokenInputNumber);

    ASSERT_EQ(token_strs.at(5), ")");
    ASSERT_EQ(tokens.at(5).token_type, TokenType::TokenRightBracket);

    ASSERT_EQ(token_strs.at(6), ",");
    ASSERT_EQ(tokens.at(6).token_type, TokenType::TokenComma);

    ASSERT_EQ(token_strs.at(7), "@1");
    ASSERT_EQ(tokens.at(7).token_type, TokenType::TokenInputNumber);

    ASSERT_EQ(token_strs.at(8), ")");
    ASSERT_EQ(tokens.at(8).token_type, TokenType::TokenRightBracket);
}

TEST(test_parser, generate_sin) {
    using namespace infer_neto;
    const std::string &str = "add(sin(@0),@1)";

    int index = 0;
    /**
          add
          /   \
        sin    @1
         |
        @0
     */
    ExpressionParser parser(str);
    parser.Tokenizer(true);
    const auto &node = parser.Generate_(index);
    ASSERT_EQ(node->num_index, int(TokenType::TokenAdd));
    ASSERT_EQ(node->left->num_index, int(TokenType::TokenSin));
    ASSERT_EQ(node->left->left->num_index, 0);
    ASSERT_EQ(node->right->num_index, 1);
}

TEST(test_parser, generate_sin2) {
    using namespace infer_neto;
    const std::string expr = "mul(@1,sin(@0))";
    ExpressionLayer layer(expr);

    std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(3, 224, 224);
    input1->Fill(2.f);  // Fill tensor with value for @0

    std::shared_ptr<Tensor<float>> input2 = std::make_shared<Tensor<float>>(3, 224, 224);
    input2->Fill(3.f);  // Fill tensor with value for @1

    std::vector<std::shared_ptr<Tensor<float>>> inputs = {input1, input2};
    std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
    outputs.at(0) = std::make_shared<Tensor<float>>(3, 224, 224);  // Prepare output tensor

    InferStatus status = layer.Forward(inputs, outputs);
    assert(status == InferStatus::kInferSuccess);  // Check that computation was successful

    float val = 2.f;
    float expected_value = std::sin(val) * 3.f;  // Calculate expected value for the output tensor elements
    std::shared_ptr<Tensor<float>> expected_output = std::make_shared<Tensor<float>>(3, 224, 224);
    expected_output->Fill(expected_value);  // Fill expected output tensor with the calculated result

    std::shared_ptr<Tensor<float>> actual_output = outputs.front();
    bool are_equal = true;
    for (uint32_t channel = 0; channel < actual_output->channels(); ++channel) {
        for (uint32_t row = 0; row < actual_output->rows(); ++row) {
            for (uint32_t col = 0; col < actual_output->cols(); ++col) {
                if (std::abs(actual_output->at(channel, row, col) - expected_output->at(channel, row, col)) > 1e-3) {
                    are_equal = false;
                    break;
                }
            }
            if (!are_equal) break;
        }
        if (!are_equal) break;
    }
    assert(are_equal);
}