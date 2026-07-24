import asyncio  # 等待能力，用于限制端到端消息接收时间
import json  # 协议编码能力，用于模拟浏览器收发标准信封
import shutil  # 测试产物清理能力，避免检查点和导出残留
import unittest  # 异步测试能力，用于启动真实本机WebSocket服务

import torch  # 权重比较能力，用于确认训练参数进入检查点
import websockets  # WebSocket客户端和临时服务能力，用于验证真实网络链路

import operations  # 产物路径能力，用于清理受控测试文件
import server  # 完整连接入口，是本组端到端测试主体
from blueprints import linearBlueprint  # 共享最小可训练蓝图


class WebSocketEndToEndTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service = await websockets.serve(server.handleConnection, "127.0.0.1", 0)  # 随机空闲端口避免与开发服务冲突
        port = self.service.sockets[0].getsockname()[1]  # 从系统分配结果读取真实端口
        self.websocket = await websockets.connect(f"ws://127.0.0.1:{port}")  # 使用真实客户端完成握手
        self.checkpointPath = operations.getArtifactPath("tests/websocket-checkpoint")  # 记录测试检查点目录
        self.exportPath = operations.getArtifactPath("tests/websocket-model.py")  # 记录测试导出文件
        self.weightsPath = self.exportPath.with_suffix(".pt")  # Python导出配套保存真实模型权重

    async def asyncTearDown(self):
        await self.websocket.close()  # 先断开客户端让连接生命周期正常收口
        self.service.close()  # 停止临时监听服务
        await self.service.wait_closed()  # 等待端口完整释放
        shutil.rmtree(self.checkpointPath, ignore_errors=True)  # 清理检查点目录
        self.exportPath.unlink(missing_ok=True)  # 清理Python导出文件
        self.weightsPath.unlink(missing_ok=True)  # 清理Python导出权重

    async def requestUntil(self, messageType, messageID, data, terminalTypes):
        await self.websocket.send(json.dumps({"type": messageType, "id": messageID, "data": data}))  # 发送浏览器同形标准请求
        messages = []  # 保存本次请求的全部流式响应
        while True:
            message = json.loads(await asyncio.wait_for(self.websocket.recv(), timeout=10))  # 每条反馈最多等待十秒
            if message.get("id") != messageID:
                continue  # 忽略热重载等无请求ID广播
            messages.append(message)  # 记录属于当前请求的反馈
            if message.get("type") in terminalTypes:
                return messages  # 收到声明终态后交给测试断言

    async def test_complete_product_command_chain(self):
        blueprint = linearBlueprint()  # 同一蓝图贯穿运行、跑分、训练和产物操作

        registryMessages = await self.requestUntil("getRegistry", "registry", {}, {"getRegistry"})  # 获取服务端节点定义
        self.assertIn("linear", registryMessages[-1]["data"]["nodes"])

        runMessages = await self.requestUntil("runBlueprint", "run", {"blueprint": blueprint}, {"blueprintComplete"})  # 执行并接收节点流
        self.assertTrue(any(message["type"] == "nodeResult" for message in runMessages))
        self.assertEqual(runMessages[-1]["data"]["status"], "succeeded")

        scoreMessages = await self.requestUntil("scoreBlueprint", "score", {"blueprint": blueprint, "score": {"runs": 1, "warmupRuns": 0}}, {"scoreComplete", "scoreError"})  # 测量完整图性能
        self.assertEqual(scoreMessages[-1]["type"], "scoreComplete")
        self.assertIn("latencyMs", scoreMessages[-1]["data"])

        trainData = {"blueprint": blueprint, "training": {"epochs": 1, "stepsPerEpoch": 1, "optimizer": "adam", "learningRate": 0.001}}  # 使用最小安全合成训练
        trainMessages = await self.requestUntil("trainBlueprint", "train", trainData, {"trainComplete", "trainError"})
        self.assertTrue(any(message["type"] == "trainProgress" for message in trainMessages))
        self.assertEqual(trainMessages[-1]["type"], "trainComplete")
        trainedSession = next(session for session in server.clientModels.values() if hasattr(session.get("model"), "parameters"))  # 忽略其他单元测试留下的假模型
        trainedParameter = next(trainedSession["model"].parameters()).detach().clone()  # 记录真实训练后参数

        saveMessages = await self.requestUntil("saveCheckpoint", "save", {"blueprint": blueprint, "path": "tests/websocket-checkpoint"}, {"checkpointSaveComplete", "checkpointError"})  # 保存模型状态
        self.assertEqual(saveMessages[-1]["type"], "checkpointSaveComplete")
        savedState = torch.load(self.checkpointPath / "state_dict.pt", map_location="cpu", weights_only=True)  # 读取协议实际写入的权重
        self.assertTrue(torch.equal(trainedParameter, next(iter(savedState.values()))))  # 检查点必须保存会话训练参数而非新随机模型

        loadMessages = await self.requestUntil("loadCheckpoint", "load", {"path": "tests/websocket-checkpoint"}, {"checkpointLoadComplete", "checkpointError"})  # 恢复模型状态和蓝图
        self.assertEqual(loadMessages[-1]["type"], "checkpointLoadComplete")
        self.assertNotIn("model", loadMessages[-1]["data"])

        exportMessages = await self.requestUntil("exportPython", "export", {"blueprint": blueprint, "path": "tests/websocket-model.py"}, {"exportComplete", "exportError"})  # 导出可运行Python模型
        self.assertEqual(exportMessages[-1]["type"], "exportComplete")

        pluginMessages = await self.requestUntil("listPlugins", "plugins", {}, {"pluginList", "pluginError"})  # 查询可信插件状态
        self.assertEqual(pluginMessages[-1]["type"], "pluginList")
        self.assertIsInstance(pluginMessages[-1]["data"], list)


if __name__ == "__main__":
    unittest.main()
