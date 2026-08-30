import unittest  # 单元测试能力，用于覆盖新增常用节点

import torch  # 张量能力，用于验证各层输入输出形状

import engine  # 导入后触发完整节点注册
import registry  # 节点创建能力，用于独立测试每类组件


class CommonNodesTest(unittest.TestCase):
    def test_pooling_modes(self):
        source = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)  # 规则网格便于同时验证普通和自适应池化
        for mode in ("max", "avg"):
            node = registry.createNode("pooling", mode, {"mode": mode, "dim": "2d", "kernel_size": "2", "stride": "2", "padding": "0"})  # 普通池化输出减半
            self.assertEqual(list(node({"x": source})["out"].shape), [1, 1, 2, 2])
        for mode in ("adaptive_max", "adaptive_avg"):
            node = registry.createNode("pooling", mode, {"mode": mode, "dim": "2d", "output_size": "1"})  # 自适应池化固定输出尺寸
            self.assertEqual(list(node({"x": source})["out"].shape), [1, 1, 1, 1])

    def test_dropout_and_embedding(self):
        dropout = registry.createNode("dropout", "dropout-1", {"p": 0.9})  # 推理模式应保持输入不变
        dropout.eval()
        values = torch.ones(3, 4)
        embedding = registry.createNode("embedding", "embedding-1", {"num_embeddings": 10, "embedding_dim": 6, "padding_idx": -1})  # 嵌入节点接受整数索引

        self.assertTrue(torch.equal(dropout({"x": values})["out"], values))
        self.assertEqual(list(embedding({"indices": torch.tensor([[1, 2]])})["out"].shape), [1, 2, 6])


if __name__ == "__main__":
    unittest.main()
