import unittest  # 单元测试能力，用于锁定前后端注册表同步契约

import export_registry  # 注册表同步指令，是本组测试主体


class RegistrySyncTest(unittest.TestCase):
    # --- 验证前端离线注册表与后端完全一致 ---
    def test_frontend_registry_matches_backend(self):
        frontendText = export_registry.frontendRegistryPath.read_text(encoding="utf-8")  # 读取前端当前离线副本

        expectedText = export_registry.getRegistryFileText()  # 由后端注册表生成期望文件内容

        self.assertEqual(frontendText, expectedText, "前端registry.json已过期，请运行: uv run python export_registry.py")  # 不一致时直接给出修复命令


if __name__ == "__main__":
    unittest.main()
