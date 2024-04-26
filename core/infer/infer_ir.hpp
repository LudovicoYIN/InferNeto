#include "pnnx/ir.h"
#include "infer_operand.hpp"
#include "infer_op.hpp"
#include <glog/logging.h>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

namespace infer_neto {

/// 计算图结构，由多个计算节点和节点之间的数据流图组成
    class RuntimeGraph {
    public:
        /**
         * 初始化计算图
         * @param param_path 计算图的结构文件
         * @param bin_path 计算图中的权重文件
         */
        RuntimeGraph(std::string param_path, std::string bin_path);

        /**
         * 设置权重文件
         * @param bin_path 权重文件路径
         */
        void set_bin_path(const std::string &bin_path);

        /**
         * 设置结构文件
         * @param param_path  结构文件路径
         */
        void set_param_path(const std::string &param_path);

        /**
         * 返回结构文件
         * @return 返回结构文件
         */
        const std::string &param_path() const;

        /**
         * 返回权重文件
         * @return 返回权重文件
         */
        const std::string &bin_path() const;

        /**
         * 计算图的初始化
         * @return 是否初始化成功
         */
        bool Init();

        const std::vector<std::shared_ptr<RuntimeOperator>> &operators() const;

        /**
         * 构建计算图
         * @param input_name 计算图输入节点的名称
         * @param output_name  计算图输出节点的名称
         */
        void Build(const std::string &input_name, const std::string &output_name);

        const std::vector<std::shared_ptr<RuntimeOperator>> &get_topo_queues() const;

        /**
       * 根据计算图中的计算节点来返回Layer
       * @param op 计算图中的计算节点
       * @return 创建成功的Layer
       */
        static std::shared_ptr<Layer> CreateLayer(
                const std::shared_ptr<RuntimeOperator> &op);

        std::vector<std::shared_ptr<Tensor<float>>> Forward(
                const std::vector<std::shared_ptr<Tensor<float>>> &inputs, bool debug);

    private:
        /**
         * 初始化kuiper infer计算图节点中的输入操作数
         * @param inputs pnnx中的输入操作数
         * @param runtime_operator 计算图节点
         */
        static void InitGraphOperatorsInput(
                const std::vector<pnnx::Operand *> &inputs,
                const std::shared_ptr<RuntimeOperator> &runtime_operator);

        /**
         * 初始化kuiper infer计算图节点中的输出操作数
         * @param outputs pnnx中的输出操作数
         * @param runtime_operator 计算图节点
         */
        static void InitGraphOperatorsOutput(
                const std::vector<pnnx::Operand *> &outputs,
                const std::shared_ptr<RuntimeOperator> &runtime_operator);

        /**
         * 初始化kuiper infer计算图中的节点属性
         * @param attrs pnnx中的节点属性
         * @param runtime_operator 计算图节点
         */
        static void
        InitGraphAttrs(const std::map<std::string, pnnx::Attribute> &attrs,
                       const std::shared_ptr<RuntimeOperator> &runtime_operator);

        /**
         * 初始化kuiper infer计算图中的节点参数
         * @param params pnnx中的参数属性
         * @param runtime_operator 计算图节点
         */
        static void
        InitGraphParams(const std::map<std::string, pnnx::Parameter> &params,
                        const std::shared_ptr<RuntimeOperator> &runtime_operator);

        void ReverseTopo(const std::shared_ptr<RuntimeOperator> &root_op);

        /**
       * 探查下一层的计算节点
       * @param current_op 当前计算节点
       * @param layer_output_data 当前节点的输出，赋予到下一层计算节点的输入张量中
       */
        static void ProbeNextLayer(
                const std::shared_ptr<RuntimeOperator> &current_op,
                const std::vector<std::shared_ptr<Tensor<float>>> &layer_output_data);

    private:
        enum class GraphState {
            NeedInit = -2,
            NeedBuild = -1,
            Complete = 0,
        };

    public:
        /**
         * 返回模型当前的状态
         * @return 返回模型当前的状态
         */
        GraphState graph_state() const;
    private:
        GraphState graph_state_ = GraphState::NeedInit;
        std::string input_name_;  /// 计算图输入节点的名称
        std::string output_name_; /// 计算图输出节点的名称
        std::string param_path_;  /// 计算图的结构文件
        std::string bin_path_;    /// 计算图的权重文件

        std::vector<std::shared_ptr<RuntimeOperator>> operators_;
        std::map<std::string, std::shared_ptr<RuntimeOperator>> operators_maps_;
        std::vector<std::shared_ptr<RuntimeOperator>> topo_operators_;

        std::unique_ptr<pnnx::Graph> graph_; /// pnnx的graph
    };

} // namespace kuiper_infer