import asyncio  # 等待能力，用于限制真实WebSocket消息接收时间
import json  # 协议编码能力，用于模拟编辑器标准信封
import os  # 路径能力，用于定位后端根目录
import re  # 文本匹配能力，用于从启动日志读取实际端口
import signal  # 信号能力，用于模拟 Ctrl+C
import sys  # 解释器路径，用于启动独立后端进程
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


@unittest.skipIf(sys.platform == "win32", "Windows 无法向子进程发送 SIGINT")
class GracefulShutdownTest(unittest.IsolatedAsyncioTestCase):
    async def test_ctrl_c_closes_connections_and_exits_cleanly(self):
        backendRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 后端根目录
        process = await asyncio.create_subprocess_exec(sys.executable, "-u", "-c", "import server; server.start(port=0)", cwd=backendRoot, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)  # 真实进程才能验证信号处理
        output = []
        try:
            port = None
            while port is None:
                line = (await asyncio.wait_for(process.stdout.readline(), timeout=120)).decode("utf-8", "replace")  # 首次导入 torch 可能较慢
                self.assertTrue(line, "".join(output))  # 进程提前退出时给出完整日志
                output.append(line)
                match = re.search(r"WebSocket服务已启动: ws://127\.0\.0\.1:(\d+)", line)
                port = int(match.group(1)) if match else None

            async with websockets.connect(f"ws://127.0.0.1:{port}") as websocket:
                await websocket.send(json.dumps({"type": "getRegistry", "id": "before-stop", "data": {}}))
                self.assertEqual(json.loads(await asyncio.wait_for(websocket.recv(), timeout=10))["id"], "before-stop")  # 服务正常工作
                process.send_signal(signal.SIGINT)  # 模拟用户按下 Ctrl+C
                await asyncio.wait_for(websocket.wait_closed(), timeout=30)
                self.assertEqual(websocket.close_code, 1001)  # 服务端主动以 going away 关闭，而不是连接直接断掉

            output.append((await asyncio.wait_for(process.stdout.read(), timeout=30)).decode("utf-8", "replace"))
            self.assertEqual(await asyncio.wait_for(process.wait(), timeout=30), 0, "".join(output))  # 正常退出码
            self.assertIn("后端已退出", "".join(output))
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()


if __name__ == "__main__":
    unittest.main()
