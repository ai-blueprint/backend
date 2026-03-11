"""
server.py - WebSocket服务器

用法：
    import server
    server.start()              # 使用默认参数启动
    server.start("0.0.0.0", 9000)  # 指定host和port启动
"""

import asyncio
import json
import websockets

import registry  # 节点注册表，提供 getAllForFrontend()
import engine  # 蓝图执行引擎，提供 run()

clients = set()  # 已连接的前端客户端集合，用 set 存储方便增删


async def sendMessage(ws, messageType, messageId, messageData):
    """
    向单个客户端发送普通消息。
    消息格式：{"type": ..., "id": ..., "data": ...}
    messageId 用于前端将响应匹配回对应的请求。
    """
    try:
        await ws.send(json.dumps({"type": messageType, "id": messageId, "data": messageData}))  # 组装消息序列化后发送给客户端
    except Exception as error:
        print(f"发送消息失败（{messageType}）: {error}")  # 客户端已断开，记录错误日志


async def sendError(ws, messageType, messageId, errorMsg):
    """
    向单个客户端发送错误消息。
    消息格式：{"type": ..., "id": ..., "error": ...}
    用 error 字段而非 data，方便前端区分正常响应和错误响应。
    """
    try:
        await ws.send(json.dumps({"type": messageType, "id": messageId, "error": errorMsg}))  # 组装序列化后发送给客户端
    except Exception as error:
        print(f"发送错误消息失败（{messageType}）: {error}")  # 客户端已断开，记录错误日志


async def broadcast(messageType, messageData):
    """
    向所有已连接客户端广播消息。
    广播消息没有 id，因为它不对应任何一个前端请求。
    发送失败的客户端会被静默跳过，不影响其他客户端。
    """
    payload = {"type": messageType, "data": messageData}  # 序列化消息体
    clientSnapshot = list(clients)  # 快照防止迭代时集合被修改
    for ws in clientSnapshot:
        try:
            await ws.send(json.dumps(payload))  # 尝试发送给这个客户端
        except Exception as error:
            print(f"广播消息失败: {error}")  # 客户端已断开，记录日志但继续发送给其他人


async def handleMessage(ws, message):
    """
    解析并分发单条前端消息。
    前端消息格式：{"type": ..., "id": ..., "data": ...}
    """
    # --- 解析消息 ---
    try:
        requestData = json.loads(message)  # 尝试将原始字符串解析成JSON
    except json.JSONDecodeError as error:
        print(f"JSON解析失败: {error}")  # 前端发来了格式错误的消息
        await sendError(ws, "parseError", "", f"无效的JSON格式: {str(error)}")  # 直接返回解析错误
        return
    # --- 提取消息元数据 ---
    messageType = requestData.get("type", "")
    messageId = requestData.get("id", "")

    # --- 分发不同类型的消息 ---
    if messageType == "getRegistry":
        try:
            result = registry.getAllForFrontend()  # 节点注册表获取异常
            await sendMessage(ws, messageType, messageId, result)  # 返回前端请求的节点列表
        except Exception as error:
            print(f"获取节点注册表失败: {error}")  # 记录获取异常
            try:
                await sendError(ws, "getRegistry", messageId, f"获取节点列表失败: {str(error)}")  # 返回错误给前端
            except Exception:
                pass  # 如果返回错误也失败了，只能放弃

    elif messageType == "runBlueprint":
        # --- 校验蓝图数据 ---
        try:
            blueprintConfig = requestData["data"]  # 尝试取出蓝图配置对象
        except (KeyError, TypeError) as error:
            await sendError(ws, "runBlueprint", messageId, "消息结构不合法，缺少data字段")  # 数据格式错误就直接拒绝
            return
        
        blueprint = blueprintConfig.get("blueprint") if blueprintConfig else None  # 从配置对象中提取蓝图
        if not blueprint:
            await sendError(ws, "runBlueprint", messageId, "缺少blueprint数据")  # 蓝图为空无法执行
            return
        
        print(f"开始执行蓝图: {blueprint}")  # 记录日志

        # --- 执行蓝图 ---
        async def onNodeResult(nodeId, nodeResult):
            try:
                await sendMessage(ws, "nodeResult", messageId, {"nodeId": nodeId, "result": nodeResult})  # 实时推送每个节点的执行结果
            except Exception as error:
                print(f"发送节点结果失败: {error}")  # 客户端可能已断开，不中断整个蓝图执行

        async def onNodeError(nodeId, nodeError):
            try:
                await sendError(ws, "nodeError", messageId, {"nodeId": nodeId, "error": nodeError})  # 实时推送节点的执行错误
            except Exception as error:
                print(f"发送节点错误失败: {error}")  # 客户端可能已断开，不中断整个蓝图执行

        try:
            blueprintResult = await engine.run(blueprint, onNodeResult, onNodeError)  # 执行蓝图（可能很耗时）
            await sendMessage(ws, "blueprintComplete", messageId, {"result": blueprintResult})  # 蓝图执行完成，返回最终结果
        except Exception as error:
            print(f"蓝图执行失败: {error}")  # 蓝图执行过程中发生了异常
            await sendError(ws, "runBlueprint", messageId, f"蓝图执行失败: {str(error)}")  # 把异常原因推送给前端

    else:
        await sendError(ws, "unknown", messageId, f"未知消息类型: {messageType}")  # 无法识别的消息类型


async def handleConnection(ws):
    """
    管理单个客户端连接的完整生命周期：加入 → 收消息 → 离开。
    """
    clients.add(ws)  # 将新连接加入客户端集合
    print(f"前端已连接，当前连接数: {len(clients)}")  # 记录连接日志

    try:
        async for message in ws:  # 持续接收客户端发来的消息
            try:
                await handleMessage(ws, message)  # 处理单条消息
            except Exception as error:
                print(f"处理消息失败: {error}")  # 一条消息处理出错就记日志，但不断开连接
                # 继续等待客户端的下一条消息
    except Exception as error:
        print(f"连接异常（可能是网络错误或协议错误）: {error}")  # 连接层发生异常
        # 异常可能导致循环提前退出
    finally:
        clients.discard(ws)  # 无论何种原因断开，都要清理这个连接的资源
        print(f"前端已断开，当前连接数: {len(clients)}")  # 记录断开日志


def start(host="0.0.0.0", port=8765):
    """
    启动 WebSocket 服务器，阻塞运行直到进程终止。
    """
    print(f"WebSocket服务启动中... ws://{host}:{port}")  # 启动前提示

    async def main():
        async with websockets.serve(handleConnection, host, port):
            print(f"WebSocket服务已启动: ws://{host}:{port}")  # 启动成功提示

            # 可拔插热重载：需要时保留下面两行，不需要时注释即可拔出
            import plugin_hot_reload  # 导入热重载模块
            plugin_hot_reload.mountHotReload(asyncio.get_running_loop(), broadcast)  # 挂载热重载机制

            await asyncio.Future()  # 永久挂起阻塞，保持服务器常驻

    asyncio.run(main())  # 启动异步事件循环
