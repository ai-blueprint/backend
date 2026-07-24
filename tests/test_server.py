import json  # 消息解析能力，用于检查假WebSocket收到的协议对象
import unittest  # 异步单元测试能力，用于覆盖WebSocket入口
from unittest.mock import patch  # 指令替换能力，用于隔离检查点文件读写

import server  # WebSocket触发入口，是本组测试主体
from blueprints import linearBlueprint  # 共享最小蓝图用于成功和失败终态测试


class FakeWebSocket:
    def __init__(self):
        self.messages = []  # 按发送顺序保存服务端JSON文本

    async def send(self, message):
        self.messages.append(json.loads(message))  # 立即解析以便断言协议字段


class ServerProtocolTest(unittest.IsolatedAsyncioTestCase):
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

    async def test_failed_run_still_sends_one_blueprint_complete(self):
        websocket = FakeWebSocket()  # 收集失败流和终态消息
        blueprint = linearBlueprint(); blueprint["nodes"][1]["data"]["params"]["in_features"] = 99  # 制造线性层输入形状错误

        await server.handleMessage(websocket, json.dumps({"type": "runBlueprint", "id": "run-error", "data": {"blueprint": blueprint}}))  # 执行失败蓝图

        terminalMessages = [message for message in websocket.messages if message["type"] == "blueprintComplete"]
        self.assertEqual(len(terminalMessages), 1)
        self.assertEqual(terminalMessages[0]["data"]["status"], "failed")
        self.assertIn("code", terminalMessages[0]["data"]["error"])

    async def test_malformed_run_uses_blueprint_terminal_envelope(self):
        websocket = FakeWebSocket()  # 畸形运行请求也必须遵守唯一终态协议

        await server.handleMessage(websocket, json.dumps({"type": "runBlueprint", "id": "bad-run", "data": {}}))  # 请求缺少蓝图对象

        self.assertEqual(len(websocket.messages), 1)
        self.assertEqual(websocket.messages[0]["type"], "blueprintComplete")
        self.assertEqual(websocket.messages[0]["data"]["status"], "failed")

    async def test_checkpoint_load_removes_internal_model_before_response(self):
        websocket = FakeWebSocket()  # 检查点恢复响应必须保持纯JSON数据
        loadedData = {"status": "complete", "path": "artifacts/checkpoints/model", "blueprint": linearBlueprint(), "manifest": {}, "model": object()}  # 模拟包含内部模型的操作结果

        with patch("server.operations.loadCheckpoint", return_value=loadedData):
            await server.handleMessage(websocket, json.dumps({"type": "loadCheckpoint", "id": "load-1", "data": {"path": "checkpoints/model"}}))  # 触发加载反馈

        self.assertEqual(websocket.messages[0]["type"], "checkpointLoadComplete")
        self.assertNotIn("model", websocket.messages[0]["data"])
        server.clientModels.pop(websocket, None)  # 假连接不经过handleConnection，测试主动释放会话模型


if __name__ == "__main__":
    unittest.main()
