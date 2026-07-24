import asyncio  # 异步运行能力，用于验证逐节点即时反馈顺序
import unittest  # 单元测试能力，用于验证持久图模型和结果协议
from unittest import mock  # 节点替换能力，用于记录计算和反馈先后顺序

import torch  # 张量能力，用于构造确定输入和检查结果

import engine  # 蓝图执行能力，是本组测试的主体
from blueprints import linearBlueprint  # 共享最小可训练蓝图，减少重复图噪声


class BlueprintModelTest(unittest.TestCase):
    def test_compiles_dotted_ids_into_persistent_module_dict(self):
        model = engine.compileBlueprint(linearBlueprint())  # 前端真实 ID 可能包含 ModuleDict 不接受的点号
        firstParameterID = id(next(model.parameters()))  # 记录首次执行前的参数对象身份

        firstOutput = model({"input.with.dot": torch.ones(2, 3)})["output-1"]  # 使用显式输入执行第一次前向
        secondOutput = model({"input.with.dot": torch.ones(2, 3)})["output-1"]  # 再次执行必须复用相同模块和参数

        self.assertIsInstance(model.nodeModules, torch.nn.ModuleDict)
        self.assertEqual(id(next(model.parameters())), firstParameterID)
        self.assertTrue(torch.equal(firstOutput, secondOutput))

    def test_input_node_keeps_random_preview_without_explicit_input(self):
        blueprint = {
            "nodes": [{"id": "input-1", "data": {"opcode": "input", "params": {"out_shape": [8]}}}, {"id": "output-1", "data": {"opcode": "output", "params": {}}}],
            "edges": [{"source": "input-1", "sourceHandle": "out", "target": "output-1", "targetHandle": "in"}],
        }  # 随机输入必须通过显式路径到达输出
        model = engine.compileBlueprint(blueprint)

        firstOutput = model()["output-1"]  # 省略输入触发随机预览
        secondOutput = model()["output-1"]  # 下一次预览应生成新值

        self.assertEqual(list(firstOutput.shape), [8])
        self.assertFalse(torch.equal(firstOutput, secondOutput))
        self.assertTrue(torch.all(firstOutput >= -1))
        self.assertTrue(torch.all(firstOutput <= 1))  # 随机输入始终限制在可视化满色范围内

    def test_all_input_reachable_branches_are_compiled_and_executed(self):
        blueprint = linearBlueprint()  # 主路径包含输入、线性层和输出
        blueprint["nodes"].extend([
            {"id": "branch-relu", "data": {"opcode": "relu", "params": {}}},
            {"id": "orphan-relu", "data": {"opcode": "relu", "params": {}}},
        ])  # 分支节点不连接Output，孤立节点没有任何输入来源
        blueprint["edges"].append({"source": "input.with.dot", "sourceHandle": "out", "target": "branch-relu", "targetHandle": "x"})  # 输入可达分支必须执行到末端
        model = engine.compileBlueprint(blueprint)  # 编译器只裁剪没有输入来源的区域
        reportedIDs = []  # 收集实际执行过的节点

        model(nodeCallback=lambda nodeID, _outputs, _duration: reportedIDs.append(nodeID))  # 使用随机输入执行全部下游

        self.assertEqual(model.sortedIDs, ["input.with.dot", "linear-1", "branch-relu", "output-1"])
        self.assertEqual(reportedIDs, model.sortedIDs)
        self.assertIn("branch-relu", model.nodeData)
        self.assertNotIn("orphan-relu", model.nodeData)

    def test_blueprint_without_output_still_executes_input_descendants(self):
        blueprint = {
            "nodes": [
                {"id": "input-1", "data": {"opcode": "input", "params": {"out_shape": [2]}}},
                {"id": "relu-1", "data": {"opcode": "relu", "params": {}}},
            ],
            "edges": [{"source": "input-1", "sourceHandle": "out", "target": "relu-1", "targetHandle": "x"}],
        }  # 实时观察不要求分支最终连接Output

        model = engine.compileBlueprint(blueprint)

        self.assertEqual(model.sortedIDs, ["input-1", "relu-1"])
        self.assertEqual(model.outputIDs, [])
        self.assertEqual(model(), {})  # 没有Output时仍完成传播，只是不产生显式模型输出

    def test_run_reports_each_node_before_computing_downstream(self):
        events = []  # 按时间保存节点计算和消息反馈动作
        blueprint = {
            "nodes": [
                {"id": "input-1", "data": {"opcode": "input", "params": {"out_shape": [2, 2]}}},
                {"id": "relu-1", "data": {"opcode": "relu", "params": {}}},
                {"id": "sigmoid-1", "data": {"opcode": "sigmoid", "params": {}}},
                {"id": "output-1", "data": {"opcode": "output", "params": {}}},
            ],
            "edges": [
                {"source": "input-1", "sourceHandle": "out", "target": "relu-1", "targetHandle": "x"},
                {"source": "relu-1", "sourceHandle": "out", "target": "sigmoid-1", "targetHandle": "x"},
                {"source": "sigmoid-1", "sourceHandle": "out", "target": "output-1", "targetHandle": "in"},
            ],
        }  # 两个连续算子用于观察上游反馈是否先于下游计算

        def computeRelu(_node, inputs):
            events.append("compute-relu")  # 记录ReLU开始计算
            return {"out": inputs["x"]}

        def computeSigmoid(_node, inputs):
            events.append("compute-sigmoid")  # 记录Sigmoid开始计算
            return {"out": inputs["x"]}

        async def onMessage(nodeID, _report):
            events.append(f"send-{nodeID}")  # 记录该节点结果已经发送

        async def onError(_nodeID, error):
            self.fail(error)  # 本测试不应进入错误路径

        reluClass = engine.registry.nodes["relu"]["cls"]  # 读取已注册ReLU节点类
        sigmoidClass = engine.registry.nodes["sigmoid"]["cls"]  # 读取已注册Sigmoid节点类
        with mock.patch.object(reluClass, "compute", computeRelu), mock.patch.object(sigmoidClass, "compute", computeSigmoid):
            asyncio.run(engine.run(blueprint, onMessage, onError))  # 执行真实异步逐节点反馈链

        self.assertLess(events.index("compute-relu"), events.index("send-relu-1"))
        self.assertLess(events.index("send-relu-1"), events.index("compute-sigmoid"))  # 上游发送完成后才能计算下游

    def test_run_continues_independent_branch_after_node_error(self):
        blueprint = linearBlueprint()  # 主路径的Linear将故意失败
        blueprint["nodes"][1]["data"]["params"]["in_features"] = 99  # 输入末维不匹配制造局部执行错误
        blueprint["nodes"].append({"id": "branch-relu", "data": {"opcode": "relu", "params": {}}})
        blueprint["edges"].append({"source": "input.with.dot", "sourceHandle": "out", "target": "branch-relu", "targetHandle": "x"})  # 独立分支仍有有效输入
        reportedIDs = []
        errorIDs = []

        async def onMessage(nodeID, _report):
            reportedIDs.append(nodeID)

        async def onError(nodeID, _error):
            errorIDs.append(nodeID)

        result = asyncio.run(engine.run(blueprint, onMessage, onError))

        self.assertEqual(reportedIDs, ["input.with.dot", "branch-relu"])  # 依赖错误Linear的Output被跳过
        self.assertEqual(errorIDs, ["linear-1"])
        self.assertEqual(result["status"], "completedWithErrors")
        self.assertEqual(result["errorCount"], 1)

    def test_serialization_is_bounded_and_describes_tensor(self):
        serialized = engine.serializeValue(torch.arange(10, dtype=torch.float32), maxValues=4)  # 大张量只预览前四个值

        self.assertEqual(serialized["shape"], [10])
        self.assertEqual(serialized["dtype"], "float32")
        self.assertEqual(serialized["device"], "cpu")
        self.assertEqual(serialized["values"], [0.0, 1.0, 2.0, 3.0])
        self.assertTrue(serialized["truncated"])

    def test_serialization_replaces_non_finite_values(self):
        serialized = engine.serializeValue(torch.tensor([float("nan"), float("inf"), float("-inf")]))  # 浏览器JSON不支持特殊浮点字面量

        self.assertEqual(serialized["values"], [None, None, None])

    def test_unknown_model_input_is_rejected(self):
        model = engine.compileBlueprint(linearBlueprint())  # 显式输入只有input.with.dot

        with self.assertRaisesRegex(engine.BlueprintError, "未知输入节点"):
            engine.validateModelInputs(model, {"misspelled": torch.zeros(2, 3)})  # 拼写错误不能静默改用随机输入


class AttentionCorrectnessTest(unittest.TestCase):
    def test_scaled_attention_returns_weights_and_respects_eval_dropout(self):
        node = engine.registry.createNode("scaled_dot_product_attention", "attention-1", {"dropout": 0.8, "is_causal": False, "scale": 0.0})  # 高失活率可确认 eval 模式关闭随机性
        node.eval()  # 推理状态不应随机丢弃权重
        query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # 两个正交查询形成可验证权重

        first = node({"q": query, "k": query, "v": query})  # 执行显式权重实现
        second = node({"q": query, "k": query, "v": query})  # eval 下重复执行结果必须一致

        self.assertEqual(list(first["attn_weights"].shape), [1, 2, 2])
        self.assertTrue(torch.allclose(first["attn_weights"].sum(dim=-1), torch.ones(1, 2)))
        self.assertTrue(torch.equal(first["out"], second["out"]))

    def test_multihead_attention_uses_batch_first_and_per_head_weights(self):
        node = engine.registry.createNode("multihead_attention", "attention-2", {"embed_dim": 4, "num_heads": 2, "dropout": 0.0, "kdim": 4, "vdim": 4})  # 使用小维度验证文档约定
        values = torch.randn(3, 5, 4)  # 输入顺序明确为 batch、sequence、feature

        result = node({"q": values, "k": values, "v": values})  # 自注意力共享三个输入

        self.assertEqual(list(result["out"].shape), [3, 5, 4])
        self.assertEqual(list(result["attn_weights"].shape), [3, 2, 5, 5])


if __name__ == "__main__":
    unittest.main()
