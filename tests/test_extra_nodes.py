import unittest  # 单元测试能力，用于锁定扩展 PyTorch 节点的关键契约

import torch  # 张量能力，用于构造可预测形状和关系目标

import engine  # 导入执行引擎以触发全部内置节点注册
import registry  # 节点创建能力，用于独立验证扩展组件


class ExtraNodesTest(unittest.TestCase):
    # --- 验证扩展节点规模和分类可见性 ---
    def test_registry_exposes_at_least_one_hundred_core_nodes(self):
        frontendRegistry = registry.getAllForFrontend()  # 隐藏示例节点后得到用户实际可见注册表

        self.assertGreaterEqual(len(frontendRegistry["nodes"]), 100)
        self.assertEqual(list(frontendRegistry["categories"]), ["base", "transform", "activation", "loss", "normalization", "shape", "math"])
        self.assertIn("time_shift", frontendRegistry["nodes"])
        self.assertIn("gaussian_nll_loss", frontendRegistry["nodes"])

    # --- 验证原生序列节点可组合出RWKV所需时间移位 ---
    def test_sequence_nodes_keep_native_tensor_semantics(self):
        values = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3).requires_grad_()  # 构造批量序列并保留梯度
        shifted = registry.createNode("time_shift", "shift", {})({"x": values})["out"]  # 时间移位首位补零并取前一位置
        sliced = registry.createNode("slice", "slice", {"dim": 1, "start": 1, "end": -1})({"x": values})["out"]  # 截取序列中间部分
        selected = registry.createNode("select", "select", {"dim": 1, "index": -1})({"x": values})["out"]  # 选择最后一个位置

        self.assertTrue(torch.equal(shifted[:, 1:], values[:, :-1]))  # 移位结果应符合RWKV时间差分语义
        self.assertEqual(list(sliced.shape), [2, 3, 3])  # 切片保持批次和特征维度
        self.assertEqual(list(selected.shape), [2, 3])  # 选择位置移除序列维度
        (shifted.sum() + sliced.sum() + selected.sum()).backward()  # 原生节点输出必须仍支持反向传播
        self.assertIsNotNone(values.grad)  # 输入应收到梯度

    # --- 验证空间重排层保持元素数量 ---
    def test_pixel_shuffle_round_trip(self):
        source = torch.randn(2, 4, 4, 8)  # 四个通道满足2倍像素上采样要求
        shuffle = registry.createNode("pixel_shuffle", "shuffle", {"upscale_factor": 2})  # 通道重排为空间
        unshuffle = registry.createNode("pixel_unshuffle", "unshuffle", {"downscale_factor": 2})  # 空间重排回通道

        enlarged = shuffle({"x": source})["out"]
        restored = unshuffle({"x": enlarged})["out"]

        self.assertEqual(list(enlarged.shape), [2, 1, 8, 16])
        self.assertTrue(torch.equal(restored, source))

    # --- 验证重复节点已合并且旧操作码仍可迁移 ---
    def test_merged_nodes_keep_legacy_blueprints_working(self):
        frontendNodes = registry.getAllForFrontend()["nodes"]  # 前端只显示新的统一节点
        for opcode in ("dropout1d", "dropout2d", "zeros_like", "ones_like", "floor", "ceil", "pixel_shuffle", "pixel_unshuffle"):
            self.assertNotIn(opcode, frontendNodes)  # 旧节点不再出现在节点列表
        for opcode in ("dropout", "tensor_like", "rounding", "pixel_rearrange"):
            self.assertIn(opcode, frontendNodes)  # 统一节点必须正常注册

        values = torch.randn(2, 4, 8)  # 使用默认三维教学张量验证旧蓝图兼容
        self.assertEqual(list(registry.createNode("zeros_like", "legacy-zero", {})({"x": values})["out"].shape), [2, 4, 8])  # 旧同形创建节点自动迁移
        self.assertEqual(list(registry.createNode("floor", "legacy-floor", {})({"x": values})["out"].shape), [2, 4, 8])  # 旧取整节点自动迁移

    # --- 验证完整 Transformer 堆栈保持教学序列契约 ---
    # --- 验证分布损失会规范化随机教学输入 ---
    def test_distribution_losses_accept_signed_inputs(self):
        prediction = torch.randn(2, 4, 8)  # 模拟默认Input节点产生的有符号张量
        target = torch.randn(2, 4, 8)  # 目标同样来自默认随机输入
        variance = torch.randn(2, 4, 8)  # 方差端口故意包含负数以验证安全转换
        klLoss = registry.createNode("kl_div_loss", "kl", {})  # KL节点内部生成合法概率分布
        gaussianLoss = registry.createNode("gaussian_nll_loss", "gaussian", {})  # 高斯节点内部保证方差为正

        klValue = klLoss({"input": prediction, "target": target})["loss"]
        gaussianValue = gaussianLoss({"input": prediction, "target": target, "variance": variance})["loss"]

        self.assertTrue(torch.isfinite(klValue).all().item())
        self.assertTrue(torch.isfinite(gaussianValue).all().item())


if __name__ == "__main__":
    unittest.main()
