//
// Created by fss on 22-11-28.
//

#ifndef INFERNETO_INFER_PARAMETER_HPP_
#define INFERNETO_INFER_PARAMETER_HPP_
#include "status_code.hpp"
#include <string>
#include <vector>

namespace infer_neto {
/**
 * 计算节点中的参数信息，参数一共可以分为如下的几类
 * 1.int
 * 2.float
 * 3.string
 * 4.bool
 * 5.int array
 * 6.string array
 * 7.float array
 */
    struct RuntimeParameter {  /// 计算节点中的参数信息
        virtual ~RuntimeParameter() = default;

        explicit RuntimeParameter(
                RuntimeParameterType type = RuntimeParameterType::kParameterUnknown)
                : type(type) {}
        RuntimeParameterType type = RuntimeParameterType::kParameterUnknown;
    };

    struct RuntimeParameterInt : public RuntimeParameter {
        RuntimeParameterInt()
                : RuntimeParameter(RuntimeParameterType::kParameterInt) {}

        explicit RuntimeParameterInt(int param_value)
                : RuntimeParameter(RuntimeParameterType::kParameterInt),
                  value(param_value) {}

        int value = 0;
    };

    struct RuntimeParameterFloat : public RuntimeParameter {
        RuntimeParameterFloat()
                : RuntimeParameter(RuntimeParameterType::kParameterFloat) {}

        explicit RuntimeParameterFloat(float param_value)
                : RuntimeParameter(RuntimeParameterType::kParameterFloat),
                  value(param_value) {}

        float value = 0.f;
    };

    struct RuntimeParameterString : public RuntimeParameter {
        RuntimeParameterString()
                : RuntimeParameter(RuntimeParameterType::kParameterString) {}

        explicit RuntimeParameterString(std::string param_value)
                : RuntimeParameter(RuntimeParameterType::kParameterString),
                  value(std::move(param_value)) {}

        std::string value;
    };

    struct RuntimeParameterIntArray : public RuntimeParameter {
        RuntimeParameterIntArray()
                : RuntimeParameter(RuntimeParameterType::kParameterIntArray) {}

        explicit RuntimeParameterIntArray(std::vector<int> param_value)
                : RuntimeParameter(RuntimeParameterType::kParameterIntArray),
                  value(std::move(param_value)) {}

        std::vector<int> value;
    };

    struct RuntimeParameterFloatArray : public RuntimeParameter {
        RuntimeParameterFloatArray()
                : RuntimeParameter(RuntimeParameterType::kParameterFloatArray) {}

        explicit RuntimeParameterFloatArray(std::vector<float> param_value)
                : RuntimeParameter(RuntimeParameterType::kParameterFloatArray),
                  value(std::move(param_value)) {}

        std::vector<float> value;
    };

    struct RuntimeParameterStringArray : public RuntimeParameter {
        RuntimeParameterStringArray()
                : RuntimeParameter(RuntimeParameterType::kParameterStringArray) {}

        explicit RuntimeParameterStringArray(std::vector<std::string> param_value)
                : RuntimeParameter(RuntimeParameterType::kParameterStringArray),
                  value(std::move(param_value)) {}

        std::vector<std::string> value;
    };

    struct RuntimeParameterBool : public RuntimeParameter {
        RuntimeParameterBool()
                : RuntimeParameter(RuntimeParameterType::kParameterBool) {}

        explicit RuntimeParameterBool(bool param_value)
                : RuntimeParameter(RuntimeParameterType::kParameterBool),
                  value(param_value) {}

        bool value = false;
    };
}
#endif //INFERNETO_INFER_PARAMETER_HPP_
