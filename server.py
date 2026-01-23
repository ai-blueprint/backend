"""
WebSocket服务器模块

负责前后端通信，接收蓝图执行请求并推送结果。
"""

import asyncio
import json
from typing import Any, Dict, Set

import websockets
import torch

import registry  # 调用 registry.get_all_for_frontend()
import engine  # 调用 engine.run()
from utils.serialization import serialize_output, serialize_all_outputs
from utils.tensor import ensure_tensor
from utils.safe import safe_get


# ==================== 全局变量 ====================

clients: Set = set()  # clients：已连接的前端列表


# ==================== 启动服务 ====================

async def start(host: str = "localhost", port: int = 8765):  # 启动服务
    """
    启动WebSocket服务器

    参数:
        host: 监听地址
        port: 监听端口
    """
    print("🚀 WebSocket服务器启动中...")  # 打印启动信息

    async with websockets.serve(handle_connection, host, port):  # 创建 WebSocket 服务
        print(f"✅ 服务器已启动：ws://{host}:{port}")
        await asyncio.Future()  # 保持运行


# ==================== 处理连接 ====================

async def handle_connection(websocket):  # 处理连接
    """
    处理单个客户端连接

    参数:
        websocket: WebSocket连接对象
    """
    clients.add(websocket)  # 将前端加入 clients
    client_addr = websocket.remote_address
    print(f"📥 新客户端连接：{client_addr}")

    try:
        async for message in websocket:  # 循环接收消息
            await handle_message(websocket, message)  # 收到消息，调用 handle_message
    except websockets.exceptions.ConnectionClosed:
        print(f"📤 客户端断开连接：{client_addr}")
    finally:
        clients.discard(websocket)  # 连接断开，从 clients 移除


# ==================== 发送响应 ====================

async def send_response(  # 发送响应
    websocket,
    msg_type: str,
    msg_id: str,
    data: Any
):
    """
    发送响应消息

    参数:
        websocket: WebSocket连接
        msg_type: 消息类型
        msg_id: 消息ID
        data: 响应数据
    """
    response = {  # 包装成 {type, id, data}
        "type": msg_type,
        "id": msg_id,
        "data": data
    }
    await websocket.send(json.dumps(response, ensure_ascii=False))  # 转 JSON 发出去


async def send_error(  # 发送错误
    websocket,
    msg_id: str,
    error_message: str
):
    """
    发送错误响应

    参数:
        websocket: WebSocket连接
        msg_id: 消息ID
        error_message: 错误信息
    """
    response = {  # 包装成 {type, id, error}
        "type": "error",
        "id": msg_id,
        "error": error_message
    }
    await websocket.send(json.dumps(response, ensure_ascii=False))  # 发出去
    print(f"❌ 发送错误：{error_message}")


# ==================== 处理消息 ====================

async def handle_message(websocket, raw_message: str):  # 处理消息
    """
    处理客户端消息

    参数:
        websocket: WebSocket连接
        raw_message: 原始消息字符串
    """
    try:
        message = json.loads(raw_message)  # 解析 JSON
    except json.JSONDecodeError:
        await send_error(websocket, "unknown", "无效的JSON格式")
        return

    msg_type = message.get("type", "")  # 提取 type
    msg_id = message.get("id", "unknown")  # 提取 id

    print(f"📨 收到请求：type={msg_type}, id={msg_id}")

    if msg_type == "get_nodes":  # 如果 type 是 get_nodes
        registry_data = registry.get_all_for_frontend()  # 调用 registry.get_all_for_frontend()
        await send_response(websocket, "registry", msg_id, registry_data)  # 发送响应

        node_count = len(safe_get(registry_data, 'nodes', default={}))
        print(f"✅ 已发送注册表，包含 {node_count} 个节点")

    elif msg_type == "run_blueprint":  # 如果 type 是 run_blueprint
        data = message.get("data", {})
        blueprint = data.get("blueprint")  # 提取 blueprint
        inputs_raw = data.get("inputs", {})  # 提取 inputs

        if not blueprint:
            await send_error(websocket, msg_id, "缺少蓝图数据")
            return

        try:
            # 准备输入数据
            initial_inputs = _prepare_inputs(inputs_raw)

            # 定义回调函数
            async def on_progress(node_id: str, output: Any):  # 回调函数：每个节点执行完就发送进度
                """节点执行完成的回调"""
                result_data = serialize_output(output)
                await send_response(
                    websocket,
                    "node_result",
                    msg_id,
                    {"nodeId": node_id, "output": result_data}
                )
                print(f"  ↳ 节点 {node_id} 执行完成")

            # 执行蓝图
            node_count = len(blueprint.get('nodes', []))
            print(f"🔄 开始执行蓝图，共 {node_count} 个节点")

            # 创建包装器以支持异步回调
            def sync_progress(node_id, output):
                """同步转异步的进度回调包装器"""
                asyncio.create_task(on_progress(node_id, output))

            result = engine.run(blueprint, initial_inputs, sync_progress)  # 调用 engine.run()，传入回调函数

            # 等待所有异步任务完成
            await asyncio.sleep(0.1)

            # 发送完成消息
            await send_response(
                websocket,
                "execution_complete",
                msg_id,
                result
            )  # 发送完成消息
            print("✅ 蓝图执行完成")

        except Exception as e:
            import traceback
            traceback.print_exc()
            await send_error(websocket, msg_id, str(e))

    else:
        await send_error(websocket, msg_id, f"未知的消息类型：{msg_type}")


# ==================== 辅助函数 ====================

def _prepare_inputs(inputs_raw: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    将原始输入转换为张量格式

    参数:
        inputs_raw: 原始输入数据

    返回:
        转换后的输入数据
    """
    initial_inputs = {}

    for node_id, ports in inputs_raw.items():
        initial_inputs[node_id] = {}
        for port_name, value in ports.items():
            tensor = ensure_tensor(value, torch.float32)
            initial_inputs[node_id][port_name] = tensor if tensor is not None else value

    return initial_inputs
