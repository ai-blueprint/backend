import json  # 消息解析能力，用于检查假WebSocket收到的协议对象
import logging  # 日志记录能力，用于验证端口探测过滤
import unittest  # 异步单元测试能力，用于覆盖WebSocket入口

from websockets.exceptions import InvalidMessage  # 握手异常类型，用于模拟空TCP探测

import server  # WebSocket触发入口，是本组测试主体
from blueprints import linearBlueprint  # 共享最小蓝图用于成功和失败终态测试


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

    async def test_scan_returns_stackable_conclusion(self):
        websocket = FakeWebSocket()  # 收集Block扫描的单次结论
        blueprint = linearBlueprint()  # 输入-同宽线性层-输出保持形状
        blueprint["nodes"][1]["data"]["params"]["out_features"] = 3  # 与输入末维一致构成同形Block

        await server.handleMessage(websocket, json.dumps({"type": "scanBlueprint", "id": "scan-1", "data": {"blueprint": blueprint}}))  # 触发静态扫描

        self.assertEqual(len(websocket.messages), 1)
        self.assertEqual(websocket.messages[0]["type"], "scanBlueprint")
        self.assertEqual(websocket.messages[0]["data"]["status"], "stackable")


if __name__ == "__main__":
    unittest.main()
