"""
server.py - 教学实验 WebSocket 服务

服务只负责节点注册表和随机输入蓝图的逐节点传播。
用法：import server; server.start()
"""

import asyncio  # 异步服务能力，用于同时处理编辑器连接和节点热重载
import json  # 标准信封编码能力，用于浏览器与后端通信
import logging  # 日志过滤能力，用于忽略握手前断开的端口探测

import websockets  # WebSocket服务能力，用于实时反馈逐节点结果
from websockets.exceptions import InvalidMessage  # 握手异常类型，用于精准过滤无请求探测

import engine  # 蓝图执行能力，提供随机输入传播和结构化结果
import registry  # 节点注册表能力，提供编辑器节点定义

clients = set()  # 保存当前编辑器连接，供节点热重载广播使用


class IncompleteHandshakeFilter(logging.Filter):
    """忽略客户端在发送HTTP请求前主动断开的无效握手。"""

    def filter(self, record):
        error = record.exc_info[1] if record.exc_info else None  # 读取日志携带的真实异常
        isEmptyProbe = isinstance(error, InvalidMessage) and "did not receive a valid HTTP request" in str(error)  # 识别端口探测直接断开
        return not isEmptyProbe  # 仅隐藏无请求探测，其他WebSocket错误继续记录


logging.getLogger("websockets.server").addFilter(IncompleteHandshakeFilter())  # 服务启动前挂载精准过滤器


# --- 向单个客户端发送正常反馈 ---
async def sendMessage(ws, messageType, messageId, messageData):
    payload = {"type": messageType, "id": messageId, "data": messageData}  # 组装标准正常信封
    await ws.send(json.dumps(payload, allow_nan=False))  # 严格JSON保证浏览器可以解析


# --- 向单个客户端发送错误反馈 ---
async def sendError(ws, messageType, messageId, errorData):
    payload = {"type": messageType, "id": messageId, "error": errorData}  # 组装标准错误信封
    await ws.send(json.dumps(payload, allow_nan=False))  # 错误消息同样保持严格JSON


# --- 广播节点注册表变化 ---
async def broadcast(messageType, messageData):
    payload = json.dumps({"type": messageType, "data": messageData}, allow_nan=False)  # 广播不对应具体请求ID
    for ws in list(clients):
        try:
            await ws.send(payload)  # 向当前连接发送热重载反馈
        except Exception:
            clients.discard(ws)  # 已断开的连接直接移出集合


# --- 将异常转换为稳定错误结构 ---
def getErrorData(error, code="requestFailed"):
    if isinstance(error, engine.BlueprintError):
        return error.toData()  # 蓝图错误保留节点和端口上下文
    return {"code": code, "message": str(error), "details": {}}  # 未分类错误使用统一结构


# --- 读取并校验请求data ---
def getRequestData(requestData):
    messageData = requestData.get("data")  # 所有业务参数位于固定data字段
    if not isinstance(messageData, dict):
        raise engine.BlueprintError("invalidMessage", "消息data必须是对象")  # 非对象不能进入业务指令
    return messageData  # 返回经过入口校验的数据


# --- 读取并校验蓝图 ---
def getBlueprint(messageData):
    blueprint = messageData.get("blueprint")  # 运行和测量共享同一字段
    if not isinstance(blueprint, dict):
        raise engine.BlueprintError("missingBlueprint", "缺少有效的blueprint对象")  # 空蓝图无法编译
    return blueprint  # 返回明确图结构


# --- 处理一条编辑器消息 ---
async def handleMessage(ws, message):
    try:
        requestData = json.loads(message)  # 原始文本先转换为协议对象
    except (json.JSONDecodeError, TypeError) as error:
        await sendError(ws, "parseError", "", {"code": "invalidJSON", "message": f"无效的JSON格式: {error}", "details": {}})  # 解析失败立即反馈
        return
    if not isinstance(requestData, dict):
        await sendError(ws, "parseError", "", {"code": "invalidMessage", "message": "消息必须是JSON对象", "details": {}})  # 数组和标量没有路由字段
        return

    messageType = requestData.get("type", "")  # 类型决定触发哪个实验指令
    messageId = str(requestData.get("id", ""))  # 请求ID关联流式结果和终态
    if not isinstance(messageType, str) or not messageType:
        await sendError(ws, "parseError", messageId, {"code": "missingType", "message": "缺少有效的消息type", "details": {}})  # 无类型无法路由
        return

    try:
        if messageType == "getRegistry":
            await sendMessage(ws, "getRegistry", messageId, registry.getAllForFrontend())  # 返回当前教学节点定义
            return

        messageData = getRequestData(requestData)  # 业务消息必须携带对象data
        if messageType == "runBlueprint":
            blueprint = getBlueprint(messageData)  # 读取本次随机输入实验蓝图

            async def onNodeResult(nodeId, nodeResult):
                await sendMessage(ws, "nodeResult", messageId, {"nodeId": nodeId, **nodeResult})  # 按传播顺序反馈节点结果

            async def onNodeError(nodeId, nodeError):
                await sendError(ws, "nodeError", messageId, {"nodeId": nodeId, **nodeError})  # 错误直接关联画布节点

            result = await engine.run(blueprint, onNodeResult, onNodeError, messageData.get("inputs"), messageData.get("maxValues", 65536))  # InputNode缺省生成指定形状随机张量
            await sendMessage(ws, "blueprintComplete", messageId, result)  # 整张蓝图只发送一次终态
            return

        if messageType == "trainStep":
            blueprint = getBlueprint(messageData)  # 单步训练使用当前完整蓝图
            model = engine.compileBlueprintCached(blueprint)  # 蓝图不变时保留权重和优化器状态
            result = model.trainStep(messageData.get("maxValues", 65536), messageData.get("optimizer", "sgd"), messageData.get("learningRate", 0.01), messageData.get("gradientClip", 0.0))  # 使用用户超参数执行一次前向、反向和权重更新
            await sendMessage(ws, "trainStep", messageId, result)  # 返回loss、预测值、目标值和节点训练快照
            return

        if messageType == "resetTraining":
            engine.clearModelCache()  # 丢弃权重和优化器状态，下次训练重新按节点ID初始化
            await sendMessage(ws, "resetTraining", messageId, {"status": "reset"})  # 告知前端训练已经回到初始状态
            return

        await sendError(ws, "unknown", messageId, {"code": "unknownMessage", "message": f"未知消息类型: {messageType}", "details": {}})  # 未知指令统一返回结构化错误
    except Exception as error:
        if messageType == "runBlueprint":
            terminal = {"status": "failed", "error": getErrorData(error), "outputNodeIds": [], "errorCount": 1, "durationMs": 0.0}  # 畸形运行也使用唯一终态收口
            await sendMessage(ws, "blueprintComplete", messageId, terminal)
            return
        await sendError(ws, messageType, messageId, getErrorData(error))  # 保留原请求类型，让对应前端命令自动收口错误状态


# --- 管理单个编辑器连接生命周期 ---
async def handleConnection(ws):
    clients.add(ws)  # 新连接加入热重载广播集合
    print(f"前端已连接，当前连接数: {len(clients)}")  # 记录实验编辑器连接状态
    try:
        async for message in ws:
            await handleMessage(ws, message)  # 每条消息独立处理，业务错误在入口内反馈
    except Exception as error:
        print(f"连接结束: {error}")  # 记录真实连接协议或网络异常
    finally:
        clients.discard(ws)  # 无论何种原因都释放连接
        print(f"前端已断开，当前连接数: {len(clients)}")  # 记录连接收口


# --- 启动本地实验服务 ---
def start(host="127.0.0.1", port=8765):
    print(f"WebSocket服务启动中... ws://{host}:{port}")  # 默认仅监听本机，避免实验接口暴露到局域网

    async def main():
        async with websockets.serve(handleConnection, host, port):
            print(f"WebSocket服务已启动: ws://{host}:{port}")  # 握手监听成功
            import plugin_hot_reload  # 节点文件热重载作为可拔插开发能力
            plugin_hot_reload.mountHotReload(asyncio.get_running_loop(), broadcast)  # 节点变化后广播新注册表
            await asyncio.Future()  # 保持服务常驻

    asyncio.run(main())  # 启动异步事件循环
