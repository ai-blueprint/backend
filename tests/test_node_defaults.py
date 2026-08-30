import unittest  # 单元测试能力，用于锁定全部节点的默认可执行契约

import torch  # 张量能力，用于构造统一的小型默认输入并检查有限输出

import engine  # 导入执行引擎以触发全部内置节点注册
import registry  # 节点注册表能力，用于逐个创建默认节点


class NodeDefaultsTest(unittest.TestCase):
    # --- 为节点端口提供统一默认输入 ---
    def createInputs(self, opcode):
        values = torch.linspace(-0.9, 0.9, 64, dtype=torch.float32).reshape(2, 4, 8)  # 避开零值并保持默认[2,4,8]契约
        ports = registry.nodes[opcode]["ports"].get("input", {})  # 读取该节点声明的全部输入端口
        inputs = {}  # 只为普通数据端口供值，可选循环状态保持未连接
        for port in ports:
            if port in {"state", "hidden", "cell"}:
                continue  # 循环状态允许缺省，由PyTorch创建零状态
            inputs[port] = values.clone()  # 每个分支使用独立张量，避免原地操作互相污染
        return inputs  # 返回可直接连接到节点的命名输入

    # --- 递归检查节点结果没有NaN或无穷值 ---
    def assertFinite(self, value, path):
        if isinstance(value, torch.Tensor):
            self.assertTrue(torch.isfinite(value).all().item(), f"{path}产生NaN或无穷值")  # 所有默认可视化结果必须有限
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                self.assertFinite(item, f"{path}[{index}]")  # 多状态输出逐项检查

    # --- 遍历全部注册节点验证默认实例和计算 ---
    def test_every_registered_node_runs_with_default_inputs(self):
        for opcode in sorted(registry.nodes):
            with self.subTest(opcode=opcode):
                node = registry.createNode(opcode, f"default-{opcode}", {})  # 空参数必须使用节点注册默认值成功构建
                node.eval()  # 教学预览使用推理模式，关闭Dropout并使用稳定归一化统计
                outputs = node(self.createInputs(opcode))  # 模拟把默认Input连接到所有必需端口
                self.assertIsInstance(outputs, dict, f"{opcode}输出必须是端口字典")
                self.assertTrue(outputs, f"{opcode}至少需要返回一个结果端口")
                for port, value in outputs.items():
                    self.assertFinite(value, f"{opcode}.{port}")  # 每个端口都必须适合前端可视化

    # --- 验证默认输入形状是全项目统一契约 ---
    def test_input_default_shape_is_small_and_shared(self):
        inputNode = registry.createNode("input", "default-input", {})  # 使用注册默认值创建输入节点

        output = inputNode({})["out"]  # 未提供显式值时生成随机教学张量

        self.assertEqual(list(output.shape), [2, 4, 8])
        self.assertTrue(torch.all(output >= -1).item())
        self.assertTrue(torch.all(output < 1).item())

    # --- 验证跨注意力真正返回独立多头权重 ---
    def test_cross_attention_uses_declared_head_count(self):
        node = registry.createNode("cross_attention", "default-cross-attention", {})  # 默认8维特征拆成2个头
        node.eval()  # 预览模式关闭注意力Dropout，权重沿键维应严格归一
        values = torch.randn(2, 4, 8)  # Q、K、V共享默认形状即可直接连接

        result = node({"q": values, "k": values, "v": values})

        self.assertEqual(list(result["out"].shape), [2, 4, 8])
        self.assertEqual(list(result["attn_weights"].shape), [2, 2, 4, 4])
        self.assertTrue(torch.allclose(result["attn_weights"].sum(dim=-1), torch.ones(2, 2, 4), atol=1e-5))

    # --- 验证矩阵节点默认同形输入和标准模式 ---
    def test_matrix_nodes_support_default_and_standard_products(self):
        sameShape = torch.randn(2, 4, 8)  # 默认模式转置第二个输入
        standardRight = torch.randn(2, 8, 4)  # 标准模式直接满足矩阵内维
        for opcode in ("matmul", "bmm"):
            with self.subTest(opcode=opcode):
                defaultNode = registry.createNode(opcode, f"default-{opcode}", {})
                standardNode = registry.createNode(opcode, f"standard-{opcode}", {"transpose_y": False})
                self.assertEqual(list(defaultNode({"x": sameShape, "y": sameShape})["out"].shape), [2, 4, 4])
                self.assertEqual(list(standardNode({"x": sameShape, "y": standardRight})["out"].shape), [2, 4, 4])

    # --- 验证分类损失可消费默认随机目标 ---
    def test_classification_losses_accept_default_connected_inputs(self):
        logits = torch.randn(2, 4, 8)  # 第1维作为四个类别
        signedTargets = torch.rand(2, 4, 8) * 2 - 1  # 模拟第二个默认Input节点
        crossEntropy = registry.createNode("cross_entropy_loss", "default-ce", {})
        binaryEntropy = registry.createNode("bce_loss", "default-bce", {})

        ceLoss = crossEntropy({"input": logits, "target": signedTargets})["loss"]  # 同形目标自动转成类别索引
        bceLoss = binaryEntropy({"input": logits, "target": signedTargets})["loss"]  # 有符号目标自动映射到[0,1]

        self.assertTrue(torch.isfinite(ceLoss).all().item())
        self.assertTrue(torch.isfinite(bceLoss).all().item())


if __name__ == "__main__":
    unittest.main()
