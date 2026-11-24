import asyncio  # 等待能力，用于限制真实WebSocket消息接收时间
import json  # 协议编码能力，用于模拟编辑器标准信封
import os  # 路径能力，用于定位后端根目录
import re  # 文本匹配能力，用于从启动日志读取实际端口
import signal  # 信号能力，用于验证嵌入时恢复处理器
import sys  # 解释器路径，用于启动独立后端进程
import unittest  # 异步测试能力，用于启动临时本机服务
from unittest import mock  # 替换信号安装器和服务工厂以控制生命周期测试

import websockets  # WebSocket客户端和服务能力，用于验证真实网络链路

import server  # 完整连接入口，是本组端到端测试主体
import engine  # 后端计算线程生命周期
from blueprints import linearBlueprint  # 共享最小随机输入蓝图
from test_training import trainingBlueprint  # 复用真实可训练蓝图验证服务重启


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


class GracefulShutdownTest(unittest.IsolatedAsyncioTestCase):
    async def test_server_can_restart_in_the_same_process(self):
        originalServe = websockets.serve  # 保留真实服务工厂
        ports = []  # 收集每次真实监听的临时端口
        stopSignals = []  # 保存每个服务实例自己的停止事件

        class ObservedServe:
            def __init__(self, *args, **kwargs):
                self.service = originalServe(*args, **kwargs)  # 构造真实WebSocket服务上下文

            async def __aenter__(self):
                service = await self.service.__aenter__()  # 启动真实WebSocket监听
                ports.append(self.service.server.sockets[0].getsockname()[1])  # 记录系统分配的端口
                return service  # 返回标准服务对象

            async def __aexit__(self, *args):
                return await self.service.__aexit__(*args)  # 复用真实服务的关闭流程

        def captureStopSignal(loop, stopSignal):
            stopSignals.append(stopSignal)  # 测试通过事件安全触发优雅关闭
            return lambda: None  # 本用例不更改进程级信号处理器

        self.addCleanup(engine.startComputeRuntime)  # 恢复计算线程以隔离后续用例
        with mock.patch.object(server, "installStopHandlers", captureStopSignal), mock.patch.object(websockets, "serve", ObservedServe):
            for serverIndex in range(2):
                serverTask = asyncio.create_task(server.serve(port=0))  # 同进程顺序启动服务实例
                await asyncio.wait_for(self._waitForCount(ports, serverIndex + 1), timeout=5)  # 等待实际端口可用
                async with websockets.connect(f"ws://127.0.0.1:{ports[serverIndex]}") as websocket:
                    await websocket.send(json.dumps({"type": "trainStep", "id": f"train-{serverIndex}", "data": {"blueprint": trainingBlueprint()}}))  # 每次启动都验证真实训练任务
                    response = json.loads(await asyncio.wait_for(websocket.recv(), timeout=5))  # 读取单步训练协议响应
                    self.assertEqual(response["type"], "trainStep")  # 服务必须接收训练请求
                    self.assertNotIn("error", response)  # 第二次启动不能残留上次停止状态
                    stopSignals[serverIndex].set()  # 正常收尾当前服务
                    await asyncio.wait_for(websocket.wait_closed(), timeout=5)  # 服务以关闭状态释放连接
                await asyncio.wait_for(serverTask, timeout=5)  # 等待计算线程退出

    async def _waitForCount(self, values, count):
        while len(values) < count:
            await asyncio.sleep(0)  # 让服务启动协程推进到监听状态

    def test_stop_handlers_are_restored_after_service(self):
        signalNumbers = (signal.SIGINT, signal.SIGTERM)  # 服务启动前已有的进程级信号处理器
        originalHandlers = {number: signal.getsignal(number) for number in signalNumbers}
        loop = asyncio.new_event_loop()  # 用独立事件循环验证绑定生命周期
        restore = server.installStopHandlers(loop, asyncio.Event())
        restore()  # 模拟服务 finally 收尾
        self.assertEqual({number: signal.getsignal(number) for number in signalNumbers}, originalHandlers)  # 嵌入应用行为不能被服务永久替换
        loop.close()  # 处理器恢复后安全销毁事件循环

    async def startBackend(self, setupCode=""):
        backendRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 后端根目录
        self.process = await asyncio.create_subprocess_exec(sys.executable, "-u", "-c", f"{setupCode}\nimport server; server.start(port=0)", cwd=backendRoot, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)  # 真实进程才能验证信号处理
        self.output = []
        while True:
            line = (await asyncio.wait_for(self.process.stdout.readline(), timeout=120)).decode("utf-8", "replace")  # 首次导入 torch 可能较慢
            self.assertTrue(line, "".join(self.output))  # 进程提前退出时给出完整日志
            self.output.append(line)
            match = re.search(r"WebSocket服务已启动: ws://127\.0\.0\.1:(\d+)", line)
            if match:
                return int(match.group(1))

    async def assertExitsCleanly(self):
        self.output.append((await asyncio.wait_for(self.process.stdout.read(), timeout=30)).decode("utf-8", "replace"))
        self.assertEqual(await asyncio.wait_for(self.process.wait(), timeout=30), 0, "".join(self.output))  # 正常退出码
        self.assertIn("后端已退出", "".join(self.output))

    async def asyncTearDown(self):
        if getattr(self, "process", None) and self.process.returncode is None:
            self.process.kill()  # 用例失败时不留下后台进程
            await self.process.wait()

    @unittest.skipIf(sys.platform == "win32", "Windows 无法向子进程发送 SIGINT")
    async def test_ctrl_c_closes_connections_and_exits_cleanly(self):
        port = await self.startBackend()
        async with websockets.connect(f"ws://127.0.0.1:{port}") as websocket:
            await websocket.send(json.dumps({"type": "getRegistry", "id": "before-stop", "data": {}}))
            self.assertEqual(json.loads(await asyncio.wait_for(websocket.recv(), timeout=10))["id"], "before-stop")  # 服务正常工作
            self.process.send_signal(signal.SIGINT)  # 模拟用户按下 Ctrl+C
            await asyncio.wait_for(websocket.wait_closed(), timeout=30)
            self.assertEqual(websocket.close_code, 1001)  # 服务端主动以 going away 关闭，而不是连接直接断掉
        await self.assertExitsCleanly()

    @unittest.skipIf(sys.platform == "win32", "Windows 无法向子进程发送 SIGINT")
    async def test_ctrl_c_with_running_and_queued_requests(self):
        slowCompile = "import engine, time\noriginalCompile = engine.compileBlueprintCached\nengine.compileBlueprintCached = lambda blueprint, *args, **kwargs: (time.sleep(3), originalCompile(blueprint, *args, **kwargs))[1]"  # 每次编译前等待3秒，制造正在计算和排队中的请求
        port = await self.startBackend(slowCompile)
        request = {"type": "runBlueprint", "data": {"blueprint": linearBlueprint()}}

        async def runUntilClosed(requestID):
            async with websockets.connect(f"ws://127.0.0.1:{port}") as websocket:
                await websocket.send(json.dumps({**request, "id": requestID}))
                messages = []
                try:
                    async for raw in websocket:
                        messages.append(json.loads(raw))
                except websockets.ConnectionClosed:
                    pass
                return messages, websocket.close_code

        running = asyncio.create_task(runUntilClosed("running"))
        await asyncio.sleep(0.5)  # 第一个请求先进入计算线程
        queued = asyncio.create_task(runUntilClosed("queued"))
        await asyncio.sleep(0.5)  # 第二个请求在计算线程中排队
        self.process.send_signal(signal.SIGINT)
        for task in (running, queued):
            messages, closeCode = await asyncio.wait_for(task, timeout=30)  # 排队请求也必须收到终态并被关闭，不能把服务挂住
            terminal = [message for message in messages if message["type"] == "blueprintComplete"]
            self.assertEqual(len(terminal), 1, messages)
            self.assertEqual(terminal[0]["data"]["error"]["code"], "serverStopping")
            self.assertEqual(closeCode, 1001)
        await self.assertExitsCleanly()


if __name__ == "__main__":
    unittest.main()
