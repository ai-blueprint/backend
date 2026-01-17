"""
WebSocket 服务器模块：负责前后端通信

通信协议说明：
=============

客户端 -> 服务端 请求格式：
--------------------------
1. 获取节点注册表：
   {"type": "get_registry", "id": "请求ID"}

2. 运行蓝图：
   {"type": "run_blueprint", "id": "请求ID", "data": {"blueprint": {...}, "inputs": {...}}}

服务端 -> 客户端 响应格式：
--------------------------
1. 注册表响应：
   {"type": "registry", "id": "请求ID", "data": {...}}

2. 单节点执行结果（实时推送）：
   {"type": "node_result", "id": "请求ID", "data": {"nodeId": "...", "output": {...}}}

3. 执行完成：
   {"type": "execution_complete", "id": "请求ID", "data": {"success": true, "results": {...}}}

4. 错误响应：
   {"type": "error", "id": "请求ID", "data": {"message": "错误信息"}}
"""

import asyncio                                                                   # 导入异步IO库，用于异步编程
import json                                                                      # 导入JSON处理库，用于消息序列化
import websockets                                                                # 导入WebSocket库，用于建立连接
import torch                                                                     # 导入PyTorch库，用于张量处理

from registry import Registry                                                    # 导入注册表类，用于获取节点配置
from engine import BlueprintEngine                                               # 导入引擎类，用于执行蓝图


class WebSocketServer:                                                           # 定义WebSocket服务器类
    """ WebSocket服务器：处理前端请求并推送执行结果 """                               # 类文档字符串

    def __init__(self, host="localhost", port=8765):                             # 构造函数，初始化服务器配置
        self.host = host                                                         # 服务器监听的主机地址
        self.port = port                                                         # 服务器监听的端口号
        self.registry = Registry()                                               # 实例化节点注册表
        self.registry.load_nodes()                                               # 加载所有节点定义
        self.clients = set()                                                     # 存储所有已连接的客户端

    async def start(self):                                                       # 启动服务器的异步方法
        """ 启动WebSocket服务器 """                                                # 方法文档字符串
        print(f"🚀 WebSocket服务器启动中...")                                       # 打印启动信息
        async with websockets.serve(self._handle_client, self.host, self.port):  # 创建WebSocket服务器
            print(f"✅ 服务器已启动：ws://{self.host}:{self.port}")                   # 打印成功信息
            await asyncio.Future()                                               # 保持服务器运行

    async def _handle_client(self, websocket):                                   # 处理客户端连接的方法
        """ 处理单个客户端的连接 """                                                 # 方法文档字符串
        self.clients.add(websocket)                                              # 将新客户端加入集合
        client_addr = websocket.remote_address                                   # 获取客户端地址
        print(f"📥 新客户端连接：{client_addr}")                                     # 打印连接信息
        
        try:                                                                     # 尝试处理客户端消息
            async for message in websocket:                                      # 持续接收客户端消息
                await self._process_message(websocket, message)                  # 处理每条消息
        except websockets.exceptions.ConnectionClosed:                           # 捕获连接关闭异常
            print(f"📤 客户端断开连接：{client_addr}")                                # 打印断开信息
        finally:                                                                 # 无论如何最终执行
            self.clients.discard(websocket)                                      # 从集合中移除客户端

    async def _process_message(self, websocket, raw_message):                    # 处理单条消息的方法
        """ 解析并处理客户端发送的消息 """                                            # 方法文档字符串
        try:                                                                     # 尝试解析消息
            message = json.loads(raw_message)                                    # 将JSON字符串解析为字典
            msg_type = message.get("type")                                       # 获取消息类型
            msg_id = message.get("id", "unknown")                                # 获取请求ID，用于响应匹配
            
            print(f"📨 收到请求：type={msg_type}, id={msg_id}")                      # 打印请求信息
            
            if msg_type == "get_registry":                                       # 如果请求获取注册表
                await self._handle_get_registry(websocket, msg_id)               # 调用注册表处理方法
            elif msg_type == "run_blueprint":                                    # 如果请求运行蓝图
                await self._handle_run_blueprint(websocket, msg_id, message.get("data", {}))
            else:                                                                # 未知消息类型
                await self._send_error(websocket, msg_id, f"未知的消息类型：{msg_type}")
                
        except json.JSONDecodeError:                                             # JSON解析失败
            await self._send_error(websocket, "unknown", "无效的JSON格式")          # 发送错误响应

    async def _handle_get_registry(self, websocket, msg_id):                     # 处理获取注册表请求
        """ 返回节点注册表给前端 """                                                  # 方法文档字符串
        registry_data = self.registry._prepare_frontend_data()                   # 获取前端格式的注册表数据
        response = {                                                             # 构造响应消息
            "type": "registry",                                                  # 响应类型
            "id": msg_id,                                                        # 对应的请求ID
            "data": registry_data                                                # 注册表数据
        }
        await websocket.send(json.dumps(response, ensure_ascii=False))           # 发送响应
        print(f"✅ 已发送注册表，包含 {len(registry_data.get('nodes', {}))} 个节点")    # 打印成功信息

    async def _handle_run_blueprint(self, websocket, msg_id, data):              # 处理运行蓝图请求
        """ 执行蓝图并实时推送每个节点的结果 """                                        # 方法文档字符串
        blueprint = data.get("blueprint")                                        # 获取蓝图数据
        inputs_raw = data.get("inputs", {})                                      # 获取初始输入（原始格式）
        
        if not blueprint:                                                        # 如果没有蓝图数据
            await self._send_error(websocket, msg_id, "缺少蓝图数据")                # 发送错误
            return                                                               # 提前返回

        try:                                                                     # 尝试执行蓝图
            # 构建初始输入数据
            initial_inputs = self._prepare_inputs(inputs_raw)                    # 将原始输入转换为张量
            
            # 创建引擎实例
            engine = BlueprintEngine(blueprint)                                  # 实例化蓝图引擎
            
            # 定义节点执行回调函数，每个节点执行完毕后调用
            async def on_node_complete(node_id, output):                         # 回调函数定义
                result_data = self._serialize_output(output)                     # 序列化输出为可JSON化格式
                response = {                                                     # 构造节点结果消息
                    "type": "node_result",                                       # 消息类型
                    "id": msg_id,                                                # 对应的请求ID
                    "data": {                                                    # 数据载荷
                        "nodeId": node_id,                                       # 节点ID
                        "output": result_data                                    # 节点输出
                    }
                }
                await websocket.send(json.dumps(response, ensure_ascii=False))   # 发送结果
                print(f"  ↳ 节点 {node_id} 执行完成")                                # 打印进度
            
            # 执行蓝图（带回调）
            print(f"🔄 开始执行蓝图，共 {len(blueprint.get('nodes', []))} 个节点")      # 打印开始信息
            results = await engine.execute_with_callback(initial_inputs, on_node_complete)
            
            # 发送执行完成消息
            final_results = self._serialize_all_outputs(results)                 # 序列化所有结果
            complete_response = {                                                # 构造完成消息
                "type": "execution_complete",                                    # 消息类型
                "id": msg_id,                                                    # 请求ID
                "data": {                                                        # 数据载荷
                    "success": True,                                             # 执行成功标记
                    "results": final_results                                     # 所有节点结果
                }
            }
            await websocket.send(json.dumps(complete_response, ensure_ascii=False))
            print(f"✅ 蓝图执行完成")                                                # 打印完成信息
            
        except Exception as e:                                                   # 捕获执行异常
            import traceback                                                     # 导入堆栈追踪
            traceback.print_exc()                                                # 打印详细错误
            await self._send_error(websocket, msg_id, str(e))                    # 发送错误响应

    def _prepare_inputs(self, inputs_raw):                                       # 准备输入数据的方法
        """ 将原始输入转换为张量格式 """                                              # 方法文档字符串
        initial_inputs = {}                                                      # 存储转换后的输入
        for node_id, ports in inputs_raw.items():                                # 遍历每个节点的输入
            initial_inputs[node_id] = {}                                         # 初始化该节点的端口字典
            for port_name, value in ports.items():                               # 遍历每个端口
                if isinstance(value, list):                                      # 如果是列表（张量数据）
                    initial_inputs[node_id][port_name] = torch.tensor(value, dtype=torch.float32)
                else:                                                            # 其他情况直接使用
                    initial_inputs[node_id][port_name] = value                   # 保持原值
        return initial_inputs                                                    # 返回转换后的输入

    def _serialize_output(self, output):                                         # 序列化单个输出的方法
        """ 将张量输出转换为可JSON序列化的格式 """                                     # 方法文档字符串
        if output is None:                                                       # 如果输出为空
            return None                                                          # 直接返回None
        if isinstance(output, dict):                                             # 如果是字典
            result = {}                                                          # 初始化结果字典
            for key, val in output.items():                                      # 遍历每个键值对
                result[key] = self._serialize_value(val)                         # 递归序列化值
            return result                                                        # 返回序列化后的字典
        return self._serialize_value(output)                                     # 直接序列化单值

    def _serialize_value(self, val):                                             # 序列化单个值的方法
        """ 将单个值转换为JSON兼容格式 """                                           # 方法文档字符串
        if isinstance(val, torch.Tensor):                                        # 如果是PyTorch张量
            return {                                                             # 返回包含形状和数据的字典
                "type": "tensor",                                                # 标识类型
                "shape": list(val.shape),                                        # 张量形状
                "data": val.tolist()                                             # 张量数据转为列表
            }
        return val                                                               # 其他类型直接返回

    def _serialize_all_outputs(self, results):                                   # 序列化所有输出的方法
        """ 将所有节点的输出转换为可序列化格式 """                                      # 方法文档字符串
        serialized = {}                                                          # 初始化结果字典
        for node_id, output in results.items():                                  # 遍历每个节点结果
            serialized[node_id] = self._serialize_output(output)                 # 序列化并保存
        return serialized                                                        # 返回序列化后的所有结果

    async def _send_error(self, websocket, msg_id, error_message):               # 发送错误消息的方法
        """ 发送错误响应给客户端 """                                                  # 方法文档字符串
        response = {                                                             # 构造错误响应
            "type": "error",                                                     # 消息类型
            "id": msg_id,                                                        # 请求ID
            "data": {"message": error_message}                                   # 错误信息
        }
        await websocket.send(json.dumps(response, ensure_ascii=False))           # 发送响应
        print(f"❌ 发送错误：{error_message}")                                       # 打印错误日志


def run_server(host="localhost", port=8765):                                     # 启动服务器的入口函数
    """ 启动WebSocket服务器的便捷函数 """                                            # 函数文档字符串
    server = WebSocketServer(host, port)                                         # 创建服务器实例
    asyncio.run(server.start())                                                  # 运行服务器


if __name__ == "__main__":                                                       # 主程序入口
    run_server()                                                                 # 启动服务器
