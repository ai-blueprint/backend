import unittest  # 单元测试能力，用于锁定前后端注册表同步契约

import export_registry  # 注册表同步指令，是本组测试主体
import engine  # 导入引擎以触发完整内置节点注册
import registry  # 读取节点命名契约


class RegistrySyncTest(unittest.TestCase):
    # --- 验证前端离线注册表与后端完全一致 ---
    def test_frontend_registry_matches_backend(self):
        frontendText = export_registry.frontendRegistryPath.read_text(encoding="utf-8")  # 读取前端当前离线副本

        expectedText = export_registry.getRegistryFileText()  # 由后端注册表生成期望文件内容

        self.assertEqual(frontendText, expectedText, "前端registry.json已过期，请运行: uv run python export_registry.py")  # 不一致时直接给出修复命令

    def test_node_opcodes_use_snake_case_and_expose_technical_name(self):
        """
        用法：直接运行本测试类即可检查注册表命名契约。
        示例：python -m unittest tests.test_registry_sync
        """
        for opcode, definition in registry.nodes.items():
            self.assertRegex(opcode, r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")  # 注册节点身份必须可预测且可跨语言使用
            self.assertEqual(definition["technicalLabel"], registry.technicalLabels.get(opcode, opcode))  # 技术名称与操作码保持稳定映射
            self.assertTrue(definition["label"], f"节点 {opcode} 必须有白话名称")  # 节点列表需要让初学者能直接理解
            self.assertTrue(definition["description"], f"节点 {opcode} 必须有行为说明")  # 悬浮说明提供学习所需的最小上下文

        self.assertIs(registry.friendlyLabels, registry.technicalLabels)  # 旧技术名称映射名继续指向同一份数据

        frontendNodes = registry.getAllForFrontend()["nodes"]  # 前端只接收规范化后的节点
        self.assertIn("leaky_relu", frontendNodes)  # 新蓝图使用规范操作码
        self.assertNotIn("leakyRelu", frontendNodes)  # 旧操作码只作为兼容别名存在
        self.assertEqual(frontendNodes["leaky_relu"]["technicalLabel"], "LeakyReLU")  # 前端可用技术名搜索节点

    def test_node_registration_rejects_non_snake_case_opcode(self):
        """
        用法：注册一个不规范操作码并确认立即失败。
        示例：registry.registerNode("InvalidOpcode", "无效节点", {}, {}, "", object)
        """
        with self.assertRaisesRegex(ValueError, "snake_case"):
            registry.registerNode("InvalidOpcode", "无效节点", {}, {}, "", object)  # 非规范身份不能进入持久化协议
        self.assertNotIn("InvalidOpcode", registry.nodes)  # 失败注册不能污染全局注册表

    def test_legacy_leaky_relu_opcode_still_resolves(self):
        """
        用法：解析历史操作码并检查迁移目标。
        示例：registry.resolveNodeAlias("leakyRelu", {})
        """
        self.assertTrue(registry.hasNode("leakyRelu"))  # 旧蓝图仍应被识别
        self.assertEqual(registry.resolveNodeAlias("leakyRelu", {})[0], "leaky_relu")  # 旧身份迁移到规范身份


if __name__ == "__main__":
    unittest.main()
