import unittest  # 单元测试能力，用于验证持久图模型和结果协议

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
        model = engine.compileBlueprint({"nodes": [{"id": "input-1", "data": {"opcode": "input", "params": {"out_shape": [8]}}}], "edges": []})  # 叶输入同时作为兼容输出

        firstOutput = model()["input-1"]  # 省略输入触发随机预览
        secondOutput = model()["input-1"]  # 下一次预览应生成新值

        self.assertEqual(list(firstOutput.shape), [8])
        self.assertFalse(torch.equal(firstOutput, secondOutput))

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
