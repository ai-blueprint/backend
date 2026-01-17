"""
WebSocket 服务器模块

负责前后端通信，接收蓝图执行请求并推送结果。

通信协议：
===========

客户端 -> 服务端：
- get_registry: 获取节点注册表
- run_blueprint: 运行蓝图

服务端 -> 客户端：
- registry: 注册表数据
- node_result: 单节点执行结果
- execution_complete: 执行完成
- error: 错误信息
"""

import asyncio
import json
from typing import Any, Callable, Dict, Optional, Set

import websockets
import torch

from registry import Registry
from engine import BlueprintEngine
from utils.serialization import serialize_output, serialize_all_outputs
from utils.tensor import ensure_tensor
from utils.safe import safe_get


class WebSocketServer:
    """
    WebSocket服务器
    
    职责：
    1. 管理客户端连接
    2. 路由消息到对应处理器
    3. 执行蓝图并推送结果
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set = set()
        
        # 初始化注册表
        self.registry = Registry()
        self.registry.load_nodes()
        
        # 消息处理器映射
        self._handlers: Dict[str, Callable] = {
            "get_registry": self._handle_get_registry,
            "run_blueprint": self._handle_run_blueprint,
        }
    
    # ==================== 服务器生命周期 ====================
    
    async def start(self):
        """启动WebSocket服务器"""
        self._log_startup()
        async with websockets.serve(self._on_client_connect, self.host, self.port):
            print(f"✅ 服务器已启动：ws://{self.host}:{self.port}")
            await asyncio.Future()  # 保持运行
    
    def _log_startup(self):
        """打印启动信息"""
        print("🚀 WebSocket服务器启动中...")
    
    # ==================== 连接管理 ====================
    
    async def _on_client_connect(self, websocket):
        """处理新客户端连接"""
        self.clients.add(websocket)
        client_addr = websocket.remote_address
        print(f"📥 新客户端连接：{client_addr}")
        
        try:
            async for message in websocket:
                await self._route_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            print(f"📤 客户端断开连接：{client_addr}")
        finally:
            self.clients.discard(websocket)
    
    # ==================== 消息路由 ====================
    
    async def _route_message(self, websocket, raw_message: str):
        """解析并路由消息到对应处理器"""
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            await self._send_error(websocket, "unknown", "无效的JSON格式")
            return
        
        msg_type = message.get("type", "")
        msg_id = message.get("id", "unknown")
        
        print(f"📨 收到请求：type={msg_type}, id={msg_id}")
        
        # 查找并执行处理器
        handler = self._handlers.get(msg_type)
        if handler:
            await handler(websocket, msg_id, message.get("data", {}))
        else:
            await self._send_error(websocket, msg_id, f"未知的消息类型：{msg_type}")
    
    # ==================== 消息处理器 ====================
    
    async def _handle_get_registry(
        self, 
        websocket, 
        msg_id: str, 
        data: Dict[str, Any]
    ):
        """处理获取注册表请求"""
        registry_data = self.registry._prepare_frontend_data()
        
        await self._send_response(websocket, "registry", msg_id, registry_data)
        
        node_count = len(safe_get(registry_data, 'nodes', default={}))
        print(f"✅ 已发送注册表，包含 {node_count} 个节点")
    
    async def _handle_run_blueprint(
        self, 
        websocket, 
        msg_id: str, 
        data: Dict[str, Any]
    ):
        """处理运行蓝图请求"""
        blueprint = data.get("blueprint")
        inputs_raw = data.get("inputs", {})
        
        if not blueprint:
            await self._send_error(websocket, msg_id, "缺少蓝图数据")
            return
        
        try:
            await self._execute_and_stream(websocket, msg_id, blueprint, inputs_raw)
        except Exception as e:
            import traceback
            traceback.print_exc()
            await self._send_error(websocket, msg_id, str(e))
    
    async def _execute_and_stream(
        self,
        websocket,
        msg_id: str,
        blueprint: Dict[str, Any],
        inputs_raw: Dict[str, Any]
    ):
        """执行蓝图并流式推送结果"""
        # 准备输入
        initial_inputs = self._prepare_inputs(inputs_raw)
        
        # 创建引擎
        engine = BlueprintEngine(blueprint)
        
        # 节点完成回调
        async def on_node_complete(node_id: str, output: Any):
            result_data = serialize_output(output)
            await self._send_response(
                websocket, 
                "node_result", 
                msg_id, 
                {"nodeId": node_id, "output": result_data}
            )
            print(f"  ↳ 节点 {node_id} 执行完成")
        
        # 执行
        node_count = len(blueprint.get('nodes', []))
        print(f"🔄 开始执行蓝图，共 {node_count} 个节点")
        
        results = await engine.execute_with_callback(initial_inputs, on_node_complete)
        
        # 发送完成消息
        final_results = serialize_all_outputs(results)
        await self._send_response(
            websocket,
            "execution_complete",
            msg_id,
            {"success": True, "results": final_results}
        )
        print("✅ 蓝图执行完成")
    
    # ==================== 输入处理 ====================
    
    def _prepare_inputs(
        self, 
        inputs_raw: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """将原始输入转换为张量格式"""
        initial_inputs = {}
        
        for node_id, ports in inputs_raw.items():
            initial_inputs[node_id] = {}
            for port_name, value in ports.items():
                tensor = ensure_tensor(value, torch.float32)
                initial_inputs[node_id][port_name] = tensor if tensor is not None else value
        
        return initial_inputs
    
    # ==================== 响应发送 ====================
    
    async def _send_response(
        self,
        websocket,
        msg_type: str,
        msg_id: str,
        data: Any
    ):
        """发送响应消息"""
        response = {
            "type": msg_type,
            "id": msg_id,
            "data": data
        }
        await websocket.send(json.dumps(response, ensure_ascii=False))
    
    async def _send_error(
        self,
        websocket,
        msg_id: str,
        error_message: str
    ):
        """发送错误响应"""
        await self._send_response(
            websocket,
            "error",
            msg_id,
            {"message": error_message}
        )
        print(f"❌ 发送错误：{error_message}")
    
    async def broadcast(self, msg_type: str, data: Any):
        """向所有客户端广播消息"""
        message = json.dumps({
            "type": msg_type,
            "data": data
        }, ensure_ascii=False)
        
        for client in self.clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                pass


# ==================== 便捷启动函数 ====================

def run_server(host: str = "localhost", port: int = 8765):
    """启动WebSocket服务器"""
    server = WebSocketServer(host, port)
    asyncio.run(server.start())


if __name__ == "__main__":
    run_server()
