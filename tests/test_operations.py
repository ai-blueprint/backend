import importlib.util  # 动态导入能力，用于验证 Python 导出文件可直接加载
import tempfile  # 临时目录能力，用于隔离测试产物
import threading  # 取消事件能力，用于验证训练安全收口
import unittest  # 单元测试能力，用于覆盖后台业务指令
from pathlib import Path  # 路径能力，用于定位临时 artifacts 子目录
from unittest import mock  # 补丁能力，用于隔离可选 ONNX 依赖

import torch  # 张量能力，用于比较检查点状态

import engine  # 结构化错误类型，用于断言导出失败反馈
import operations  # 跑分、训练、检查点和导出指令
from blueprints import linearBlueprint  # 共享最小可训练蓝图


class OperationsTest(unittest.TestCase):
    def setUp(self):
        operations.artifactsRoot.mkdir(parents=True, exist_ok=True)  # 临时目录必须位于真实 artifacts 下以验证导出定位
        self.tempDirectory = tempfile.TemporaryDirectory(dir=operations.artifactsRoot)  # 每个测试拥有独立产物根
        self.artifactsPatch = mock.patch.object(operations, "artifactsRoot", Path(self.tempDirectory.name))  # 业务路径解析限制在临时根
        self.artifactsPatch.start()

    def tearDown(self):
        self.artifactsPatch.stop()  # 恢复真实 artifacts 根，避免影响其他测试
        self.tempDirectory.cleanup()  # 删除本测试创建的所有产物

    def test_score_reports_latency_and_parameter_counts(self):
        result = operations.scoreBlueprint(linearBlueprint(), {"warmupRuns": 0, "runs": 2, "inputs": {"input.with.dot": [[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]]}})  # 固定输入执行两次测量

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["latencyMs"]["runs"], 2)
        self.assertEqual(result["parameters"]["total"], 8)
        self.assertEqual(result["parameters"]["trainable"], 8)

    def test_training_reports_progress_and_supports_cancellation(self):
        progress = []  # 收集同步线程回调数据
        result = operations.trainBlueprint(linearBlueprint(), {"epochs": 2, "stepsPerEpoch": 2, "learningRate": 0.01}, progress.append)  # 默认合成零目标完成端到端训练
        cancelEvent = threading.Event(); cancelEvent.set()  # 预先取消可验证第一个批次边界收口
        cancelled = operations.trainBlueprint(linearBlueprint(), {"epochs": 2, "stepsPerEpoch": 2}, cancelEvent=cancelEvent)  # 不应执行任何优化步骤

        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(progress), 4)
        self.assertIsInstance(result["loss"], float)
        self.assertEqual(cancelled["status"], "cancelled")

    def test_training_accepts_frontend_parameter_objects(self):
        blueprint = linearBlueprint()  # 模拟编辑器保存的参数定义对象
        blueprint["nodes"][0]["data"]["params"]["out_shape"] = {"type": "list", "value": [2, 3], "label": "输出形状"}  # 前端节点参数保留元数据

        result = operations.trainBlueprint(blueprint, {"epochs": 1, "stepsPerEpoch": 1})  # 训练应读取编译后已解包形状

        self.assertEqual(result["status"], "complete")

    def test_checkpoint_round_trip_restores_state(self):
        blueprint = linearBlueprint()
        model = engine.compileBlueprint(blueprint)  # 创建明确状态用于保存
        with torch.no_grad():
            next(model.parameters()).fill_(2.5)  # 修改参数以区分默认初始化

        saved = operations.saveCheckpoint(blueprint, "checkpoints/demo", model, {"test": True})  # 保存清单、蓝图和 state_dict
        loaded = operations.loadCheckpoint("checkpoints/demo")  # 从受控目录重建模型并严格加载权重

        self.assertEqual(saved["manifest"]["formatVersion"], 1)
        self.assertEqual(loaded["manifest"]["metadata"], {"test": True})
        self.assertTrue(torch.equal(next(model.parameters()), next(loaded["model"].parameters())))

    def test_artifact_path_rejects_directory_traversal(self):
        with self.assertRaisesRegex(engine.BlueprintError, "artifacts"):
            operations.getArtifactPath("../outside.pt")  # 上级路径不能越过当前测试产物根

    def test_python_export_loads_without_extra_dependencies(self):
        result = operations.exportPython(linearBlueprint(), "exports/demo")  # 扩展名由命令自动补全
        exportPath = operations.backendRoot / result["path"]  # 协议返回后端相对路径
        specification = importlib.util.spec_from_file_location("exported_blueprint", exportPath)  # 模拟用户直接导入导出模块
        module = importlib.util.module_from_spec(specification); specification.loader.exec_module(module)  # 文件应仅依赖已有后端和 torch

        exportedModel = module.createModel()  # 导出入口创建完整 BlueprintModel
        self.assertEqual(exportedModel.inputIDs, ["input.with.dot"])
        self.assertTrue((exportPath.with_suffix(".pt")).is_file())

    def test_onnx_missing_dependency_returns_clear_error(self):
        realImport = __import__  # 保留其他模块的正常导入行为

        def importWithoutONNX(name, *arguments, **keywords):
            if name == "onnx":
                raise ImportError("missing for test")  # 只模拟可选 ONNX 包缺失
            return realImport(name, *arguments, **keywords)

        with mock.patch("builtins.__import__", side_effect=importWithoutONNX):
            with self.assertRaisesRegex(engine.BlueprintError, "uv sync --extra onnx") as errorContext:
                operations.exportONNX(linearBlueprint(), "exports/demo.onnx")  # 缺依赖时不得进入实际导出
        self.assertEqual(errorContext.exception.code, "exportDependencyMissing")

    def test_onnx_export_creates_valid_model_when_dependency_is_installed(self):
        try:
            import onnx  # 可选依赖存在时验证真实导出文件
        except ImportError:
            self.skipTest("未安装ONNX可选依赖")  # 基础安装仍允许运行其他测试

        result = operations.exportONNX(linearBlueprint(), "exports/verified.onnx")  # 使用输入节点默认形状完成真实追踪
        exportPath = operations.backendRoot / result["path"]  # 协议返回后端相对路径
        exportedModel = onnx.load(exportPath)  # 读取生成的二进制模型
        onnx.checker.check_model(exportedModel)  # 官方检查器验证图和权重结构

        self.assertTrue(exportPath.is_file())


if __name__ == "__main__":
    unittest.main()
