import asyncio  # 等待能力，用于限制真实WebSocket消息接收时间
import json  # 协议编码能力，用于模拟编辑器标准信封
import unittest  # 异步测试能力，用于启动临时本机服务

import websockets  # WebSocket客户端和服务能力，用于验证真实网络链路

import server  # 完整连接入口，是本组端到端测试主体
from blueprints import linearBlueprint  # 共享最小随机输入蓝图


class WebSocketEndToEndTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service = await websockets.serve(server.handleConnection, "127.0.0.1", 0)  # 使用随机空闲端口避免冲突
        port = self.service.sockets[0].getsockname()[1]  # 读取系统实际分配端口
        self.websocket = await websockets.connect(f"ws://127.0.0.1:{port}")  # 完成真实WebSocket握手

    async def asyncTearDown(self):
        await self.websocket.close()  # 先关闭客户端连接
        self.service.close()  # 停止临时服务
        await self.service.wait_closed()  # 等待端口释放

    async def requestUntil(self, messageType, messageID, data, terminalTypes):
        await self.websocket.send(json.dumps({"type": messageType, "id": messageID, "data": data}))  # 发送编辑器同形请求
        messages = []  # 保存当前请求全部流式反馈
        while True:
            message = json.loads(await asyncio.wait_for(self.websocket.recv(), timeout=10))  # 每条反馈最多等待十秒
            if message.get("id") != messageID:
                continue  # 忽略节点热重载广播
            messages.append(message)  # 记录当前请求反馈
            if message.get("type") in terminalTypes:
                return messages  # 收到终态后返回完整消息流

    async def test_random_input_experiment_chain(self):
        blueprint = linearBlueprint()  # 同一蓝图贯穿注册表、运行和传播测量

        registryMessages = await self.requestUntil("getRegistry", "registry", {}, {"getRegistry"})
        self.assertIn("linear", registryMessages[-1]["data"]["nodes"])

        runMessages = await self.requestUntil("runBlueprint", "run", {"blueprint": blueprint}, {"blueprintComplete"})
        self.assertTrue(any(message["type"] == "nodeResult" for message in runMessages))
        self.assertEqual(runMessages[-1]["data"]["status"], "succeeded")

        unsupported = await self.requestUntil("trainBlueprint", "training", {"blueprint": blueprint}, {"unknown"})  # 工程训练接口必须明确不可用
        self.assertEqual(unsupported[-1]["error"]["code"], "unknownMessage")


if __name__ == "__main__":
    unittest.main()
