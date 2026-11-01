import asyncio  # 并发能力，用于同时发出慢计算和注册表请求
import json  # 消息解析能力，用于检查假WebSocket收到的协议对象
import logging  # 日志记录能力，用于验证端口探测过滤
import tempfile  # 临时目录能力，用于隔离插件根目录
import threading  # 线程同步能力，用于精确控制慢计算何时结束
import unittest  # 异步单元测试能力，用于覆盖WebSocket入口
from pathlib import Path  # 路径能力，用于替换插件根目录
from unittest import mock  # 替换能力，用于把编译变成可控的慢操作

from websockets.exceptions import InvalidMessage  # 握手异常类型，用于模拟空TCP探测

import engine  # 执行引擎，用于替换编译函数和清理模型缓存
import plugin_hot_reload  # 热重载入口，用于验证计算线程中的注册表重建
import plugins  # 插件加载器，用于隔离热重载用例的插件目录
import registry  # 注册表，用于检查发布的前端快照
import server  # WebSocket触发入口，是本组测试主体
from blueprints import linearBlueprint  # 共享最小蓝图用于成功和失败终态测试
from test_training import trainingBlueprint  # 复用训练用例的预测加目标蓝图


class FakeWebSocket:
    def __init__(self):
        self.messages = []  # 按发送顺序保存服务端JSON文本

    async def send(self, message):
        self.messages.append(json.loads(message))  # 立即解析以便断言协议字段


class ServerProtocolTest(unittest.IsolatedAsyncioTestCase):
    async def test_empty_tcp_probe_handshake_log_is_filtered(self):
        error = InvalidMessage("did not receive a valid HTTP request")  # 模拟连接后未发送HTTP请求就断开的探测
        record = logging.LogRecord("websockets.server", logging.ERROR, __file__, 1, "opening handshake failed", (), (InvalidMessage, error, None))  # 构造服务端同形日志

        self.assertFalse(server.IncompleteHandshakeFilter().filter(record))  # 该噪声不再输出完整traceback

    async def test_malformed_inputs_return_structured_errors(self):
        websocket = FakeWebSocket()  # 不建立真实网络连接即可验证入口反馈

        await server.handleMessage(websocket, "not-json")  # 非JSON文本触发解析错误
        await server.handleMessage(websocket, "[]")  # 非对象JSON触发消息结构错误
        await server.handleMessage(websocket, json.dumps({"id": "missing-type"}))  # 缺失类型触发路由错误

        self.assertEqual([message["type"] for message in websocket.messages], ["parseError", "parseError", "parseError"])
        self.assertTrue(all(isinstance(message["error"], dict) for message in websocket.messages))
        self.assertTrue(all("code" in message["error"] for message in websocket.messages))

    async def test_run_sends_all_ports_and_exactly_one_terminal_status(self):
        websocket = FakeWebSocket()  # 收集预览的流式和终态消息
        request = {"type": "runBlueprint", "id": "run-1", "data": {"blueprint": linearBlueprint(), "inputs": {"input.with.dot": [[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]]}, "maxValues": 2}}  # 有界显式输入预览

        await server.handleMessage(websocket, json.dumps(request))  # 执行完整蓝图协议

        nodeMessages = [message for message in websocket.messages if message["type"] == "nodeResult"]
        terminalMessages = [message for message in websocket.messages if message["type"] == "blueprintComplete"]
        self.assertEqual(len(nodeMessages), 3)
        self.assertTrue(all("outputs" in message["data"] and "opcode" in message["data"] and "durationMs" in message["data"] for message in nodeMessages))
        self.assertEqual(len(terminalMessages), 1)
        self.assertEqual(terminalMessages[0]["data"]["status"], "succeeded")
        self.assertEqual(terminalMessages[0]["data"]["errorCount"], 0)

    async def test_run_reports_branch_that_does_not_reach_output(self):
        websocket = FakeWebSocket()  # 收集带半途结束分支蓝图的实际反馈
        blueprint = linearBlueprint()  # 主路径包含三个有效节点
        blueprint["nodes"].extend([
            {"id": "branch-relu", "data": {"opcode": "relu", "params": {}}},
            {"id": "orphan-relu", "data": {"opcode": "relu", "params": {}}},
        ])  # 分支有输入值，孤立节点没有输入值
        blueprint["edges"].append({"source": "input.with.dot", "sourceHandle": "out", "target": "branch-relu", "targetHandle": "x"})  # 分支末端不接Output

        await server.handleMessage(websocket, json.dumps({"type": "runBlueprint", "id": "branch-run", "data": {"blueprint": blueprint}}))  # 执行带半途结束分支的蓝图

        reportedIDs = [message["data"]["nodeId"] for message in websocket.messages if message["type"] == "nodeResult"]
        self.assertEqual(reportedIDs, ["input.with.dot", "linear-1", "branch-relu", "output-1"])
        self.assertNotIn("orphan-relu", reportedIDs)

    async def test_node_error_does_not_stop_independent_branch(self):
        websocket = FakeWebSocket()  # 收集局部错误、独立分支结果和终态消息
        blueprint = linearBlueprint(); blueprint["nodes"][1]["data"]["params"]["in_features"] = 99  # 制造线性层输入形状错误
        blueprint["nodes"].append({"id": "branch-relu", "data": {"opcode": "relu", "params": {}}})
        blueprint["edges"].append({"source": "input.with.dot", "sourceHandle": "out", "target": "branch-relu", "targetHandle": "x"})  # 不依赖错误节点的分支应继续

        await server.handleMessage(websocket, json.dumps({"type": "runBlueprint", "id": "run-error", "data": {"blueprint": blueprint}}))  # 执行包含局部错误的蓝图

        reportedIDs = [message["data"]["nodeId"] for message in websocket.messages if message["type"] == "nodeResult"]
        nodeErrors = [message["error"]["nodeId"] for message in websocket.messages if message["type"] == "nodeError"]
        terminalMessages = [message for message in websocket.messages if message["type"] == "blueprintComplete"]
        self.assertEqual(reportedIDs, ["input.with.dot", "branch-relu"])
        self.assertEqual(nodeErrors, ["linear-1"])
        self.assertEqual(len(terminalMessages), 1)
        self.assertEqual(terminalMessages[0]["data"]["status"], "completedWithErrors")
        self.assertEqual(terminalMessages[0]["data"]["errorCount"], 1)

    async def test_malformed_run_uses_blueprint_terminal_envelope(self):
        websocket = FakeWebSocket()  # 畸形运行请求也必须遵守唯一终态协议

        await server.handleMessage(websocket, json.dumps({"type": "runBlueprint", "id": "bad-run", "data": {}}))  # 请求缺少蓝图对象

        self.assertEqual(len(websocket.messages), 1)
        self.assertEqual(websocket.messages[0]["type"], "blueprintComplete")
        self.assertEqual(websocket.messages[0]["data"]["status"], "failed")

    async def test_slow_compute_does_not_block_other_requests(self):
        for messageType, data in (("runBlueprint", {"blueprint": linearBlueprint()}), ("trainStep", {"blueprint": trainingBlueprint(), "maxValues": 8})):
            with self.subTest(messageType=messageType):
                engine.clearModelCache()  # 强制本轮重新编译，进入被替换的慢编译
                computeWebSocket = FakeWebSocket()  # 收集慢计算请求的反馈
                registryWebSocket = FakeWebSocket()  # 另一个编辑器连接
                releaseCompile = threading.Event()  # 由测试决定慢编译何时结束
                compileStarted = threading.Event()  # 确认慢编译已经占住计算线程
                originalCompile = engine.compileBlueprintCached

                def slowCompile(blueprint):
                    compileStarted.set()
                    releaseCompile.wait(timeout=10)  # 模拟长时间编译或计算
                    return originalCompile(blueprint)

                with mock.patch.object(engine, "compileBlueprintCached", slowCompile):
                    computeTask = asyncio.create_task(server.handleMessage(computeWebSocket, json.dumps({"type": messageType, "id": "slow", "data": data})))
                    try:
                        await asyncio.to_thread(compileStarted.wait, 10)  # 等待计算线程真正开始执行
                        await asyncio.wait_for(server.handleMessage(registryWebSocket, json.dumps({"type": "getRegistry", "id": "registry-1"})), timeout=2)  # 计算进行中注册表请求必须立即返回
                        self.assertFalse(computeTask.done())  # 慢计算此时确实仍在进行
                    finally:
                        releaseCompile.set()  # 断言失败也要放行计算线程，避免后续用例排队超时
                        await asyncio.wait_for(computeTask, timeout=10)

                self.assertEqual(registryWebSocket.messages[0]["type"], "getRegistry")
                self.assertIn("nodes", registryWebSocket.messages[0]["data"])
                self.assertEqual(computeWebSocket.messages[-1]["type"], "blueprintComplete" if messageType == "runBlueprint" else "trainStep")
                self.assertNotIn("error", computeWebSocket.messages[-1])

    async def test_hot_reload_rebuilds_registry_in_compute_thread(self):
        statusSnapshot = dict(plugins.pluginStatuses)  # 热重载会改写插件状态，用例结束后恢复
        try:
            with tempfile.TemporaryDirectory() as emptyPluginsRoot, mock.patch.object(plugins, "pluginsRoot", Path(emptyPluginsRoot)):  # 不加载开发者本机插件，结果与环境无关
                messageType, messageData = await engine.runInWorker(plugin_hot_reload.reloadRegistry)  # 与真实热重载一样在计算线程执行
            self.assertEqual(messageType, "registryUpdated")
            self.assertIs(registry.getPublishedForFrontend(), messageData)  # 事件循环读取的是重建完成后整体发布的快照
            self.assertIn("linear", messageData["nodes"])
        finally:
            registry.publishedFrontend = None  # 恢复为直接读取注册表，避免影响其他用例
            plugins.pluginStatuses.clear(); plugins.pluginStatuses.update(statusSnapshot)

    async def test_stop_signal_aborts_run_and_training(self):
        engine.clearModelCache()  # 从编译阶段开始验证检查点
        try:
            engine.requestStop()  # 模拟 Ctrl+C 已经发出停止信号
            runWebSocket = FakeWebSocket()
            await server.handleMessage(runWebSocket, json.dumps({"type": "runBlueprint", "id": "stop-run", "data": {"blueprint": linearBlueprint()}}))
            terminal = runWebSocket.messages[-1]
            self.assertEqual(terminal["type"], "blueprintComplete")  # 中止后仍然只给前端一个终态
            self.assertEqual(terminal["data"]["status"], "failed")
            self.assertEqual(terminal["data"]["error"]["code"], "serverStopping")

            trainWebSocket = FakeWebSocket()
            await server.handleMessage(trainWebSocket, json.dumps({"type": "trainStep", "id": "stop-train", "data": {"blueprint": trainingBlueprint()}}))
            self.assertEqual(trainWebSocket.messages[-1]["type"], "trainStep")
            self.assertEqual(trainWebSocket.messages[-1]["error"]["code"], "serverStopping")
        finally:
            engine.stopEvent.clear()  # 恢复服务状态，避免影响其他用例


if __name__ == "__main__":
    unittest.main()
