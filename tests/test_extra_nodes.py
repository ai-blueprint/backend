import unittest  # 单元测试能力，用于锁定扩展 PyTorch 节点的关键契约

import torch  # 张量能力，用于构造可预测形状和关系目标

import engine  # 导入执行引擎以触发全部内置节点注册
import registry  # 节点创建能力，用于独立验证扩展组件


class ExtraNodesTest(unittest.TestCase):
    # --- 验证扩展节点规模和分类可见性 ---
    def test_registry_exposes_at_least_one_hundred_core_nodes(self):
        frontendRegistry = registry.getAllForFrontend()  # 隐藏示例节点后得到用户实际可见注册表

        self.assertGreaterEqual(len(frontendRegistry["nodes"]), 100)
        self.assertIn("activation_extra", frontendRegistry["categories"])
        self.assertIn("layers_extra", frontendRegistry["categories"])
        self.assertIn("loss_extra", frontendRegistry["categories"])
        self.assertIn("transformer_encoder", frontendRegistry["nodes"])
        self.assertIn("gaussian_nll_loss", frontendRegistry["nodes"])

    # --- 验证空间重排层保持元素数量 ---
    def test_pixel_shuffle_round_trip(self):
        source = torch.randn(2, 4, 4, 8)  # 四个通道满足2倍像素上采样要求
        shuffle = registry.createNode("pixel_shuffle", "shuffle", {"upscale_factor": 2})  # 通道重排为空间
        unshuffle = registry.createNode("pixel_unshuffle", "unshuffle", {"downscale_factor": 2})  # 空间重排回通道

        enlarged = shuffle({"x": source})["out"]
        restored = unshuffle({"x": enlarged})["out"]

        self.assertEqual(list(enlarged.shape), [2, 1, 8, 16])
        self.assertTrue(torch.equal(restored, source))

    # --- 验证完整 Transformer 堆栈保持教学序列契约 ---
    def test_transformer_stacks_keep_batch_first_shape(self):
        params = {"d_model": 8, "nhead": 2, "num_layers": 2, "dim_feedforward": 16, "dropout": 0.0}  # 小型两层堆栈保持测试快速
        encoder = registry.createNode("transformer_encoder", "encoder", params)  # 构建完整编码器而不是单层
        decoder = registry.createNode("transformer_decoder", "decoder", params)  # 构建完整解码器而不是单层
        values = torch.randn(2, 4, 8)  # 使用全项目统一批优先序列形状

        memory = encoder({"x": values})["out"]
        output = decoder({"x": values, "memory": memory})["out"]

        self.assertEqual(list(memory.shape), [2, 4, 8])
        self.assertEqual(list(output.shape), [2, 4, 8])

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
