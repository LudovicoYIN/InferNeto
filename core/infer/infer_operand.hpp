//
// Created by fss on 22-11-28.
//

#ifndef INFERNETO_INFER_OPERAND_HPP_
#define INFERNETO_INFER_OPERAND_HPP_
#include <vector>
#include <string>
#include <memory>
#include "status_code.hpp"
#include "infer_datatype.hpp"
#include "data/cpu/tensor.hpp"

namespace infer_neto {
/// 计算节点输入输出的操作数
struct RuntimeOperand {
  std::string name;                                     /// 操作数的名称
  std::vector<int32_t> shapes;                          /// 操作数的形状
  std::vector<std::shared_ptr<Tensor<float>>> datas;    /// 存储操作数
  RuntimeDataType type = RuntimeDataType::kTypeUnknown; /// 操作数的类型，一般是float
};
}
#endif //INFERNETO_INFER_OPERAND_HPP_
