import json  # 清单写入能力，用于构造本地测试插件
import copy  # 注册表快照能力，用于测试结束后恢复全局状态
import tempfile  # 临时目录能力，用于隔离可信插件根
import unittest  # 单元测试能力，用于覆盖插件生命周期
from pathlib import Path  # 路径能力，用于创建插件目录结构
from unittest import mock  # 补丁能力，用于替换插件根目录

import engine  # 导入后确保内置注册表已加载
import plugins  # 插件发现和重载指令，是本组测试主体
import registry  # 注册表数据，用于验证所有权和回滚


class PluginTest(unittest.TestCase):
    def setUp(self):
        self.tempDirectory = tempfile.TemporaryDirectory()  # 每个测试拥有独立可信根
        self.rootPatch = mock.patch.object(plugins, "pluginsRoot", Path(self.tempDirectory.name))  # 插件只能从该临时根发现
        self.rootPatch.start()
        self.registrySnapshot = (copy.copy(registry.nodes), copy.deepcopy(registry.categories), copy.copy(registry.nodeOwners), copy.copy(registry.categoryOwners))  # 保存测试前内置注册状态

    def tearDown(self):
        registry.nodes.clear(); registry.nodes.update(self.registrySnapshot[0])  # 恢复节点定义避免插件泄漏到其他测试
        registry.categories.clear(); registry.categories.update(self.registrySnapshot[1])  # 恢复分类和原始节点顺序
        registry.nodeOwners.clear(); registry.nodeOwners.update(self.registrySnapshot[2])  # 恢复节点所有权
        registry.categoryOwners.clear(); registry.categoryOwners.update(self.registrySnapshot[3])  # 恢复分类所有权
        plugins.pluginStatuses.clear()  # 清除本次临时插件的状态反馈
        self.rootPatch.stop()  # 恢复真实插件根
        self.tempDirectory.cleanup()  # 删除测试插件文件

    def createPlugin(self, pluginID, opcode):
        pluginPath = Path(self.tempDirectory.name) / pluginID; pluginPath.mkdir()  # 一级目录名就是插件身份
        manifest = {"id": pluginID, "name": pluginID, "version": "1.0.0", "entry": "plugin.py", "enabled": True}  # 写入最小有效清单
        (pluginPath / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        source = f'''from registry import BaseNode, category, node\ncategory(id="{pluginID}_category", label="Plugin", color="#000", icon="")\n@node(opcode="{opcode}", label="Plugin", ports={{"input": {{}}, "output": {{"out": ""}}}}, params={{}}, description="test")\nclass PluginNode(BaseNode):\n    def compute(self, input):\n        return {{"out": 1}}\n'''
        (pluginPath / "plugin.py").write_text(source, encoding="utf-8")  # 入口注册一个明确归属节点

    def test_reload_registers_plugin_owner(self):
        self.createPlugin("trusted", "trusted_node")  # 创建无碰撞插件

        result = plugins.reloadPlugins()  # 从可信根发现并加载插件

        self.assertEqual(result["status"], "complete")
        self.assertEqual(registry.nodeOwners["trusted_node"], "plugin:trusted")
        self.assertEqual(plugins.pluginStatuses["trusted"]["nodes"], ["trusted_node"])

    def test_collision_rolls_back_registry(self):
        self.createPlugin("collision", "linear")  # 尝试覆盖内置全连接节点
        originalLinear = registry.nodes["linear"]["cls"]  # 保存碰撞前真实实现

        with self.assertRaisesRegex(ValueError, "已由 core 注册"):
            plugins.reloadPlugins()  # 碰撞必须失败并完整回滚

        self.assertIs(registry.nodes["linear"]["cls"], originalLinear)
        self.assertEqual(registry.nodeOwners["linear"], "core")


if __name__ == "__main__":
    unittest.main()
