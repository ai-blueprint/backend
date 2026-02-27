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


async def sendMessage(ws, type, id, data):
    """
    向单个客户端发送普通消息。
    消息格式：{"type": ..., "id": ..., "data": ...}
    id 用于前端将响应匹配回对应的请求。
    """
    print(f"发送消息: {type} {data}")
    await ws.send(json.dumps({"type": type, "id": id, "data": data}))


async def sendError(ws, type, id, error):
    """
    向单个客户端发送错误消息。
    消息格式：{"type": ..., "id": ..., "error": ...}
    用 error 字段而非 data，方便前端区分正常响应和错误响应。
    """
    await ws.send(json.dumps({"type": type, "id": id, "error": error}))


async def broadcast(type, data):
    """
    向所有已连接客户端广播消息。
    广播消息没有 id，因为它不对应任何一个前端请求。
    发送失败的客户端会被静默跳过，不影响其他客户端。
    """
    msg = json.dumps({"type": type, "data": data})
    for ws in list(clients):  # list() 防止迭代时集合被修改
        try:
            await ws.send(msg)
        except Exception:
            pass  # 客户端已断开，忽略，等 handleConnection 的 finally 清理


async def handleMessage(ws, message):
    """
    解析并分发单条前端消息。
    前端消息格式：{"type": ..., "id": ..., "data": ...}
    """
    data = json.loads(message)
    msg_type = data.get("type", "")
    id = data.get("id", "")

    if msg_type == "getRegistry":
        # 前端请求节点注册表，返回过滤后的标准结构
        result = registry.getAllForFrontend()
        await sendMessage(ws, msg_type, id, result)

    elif msg_type == "runBlueprint":
        blueprint = data["data"].get("blueprint")
        if not blueprint:
            await sendError(ws, "runBlueprint", id, "缺少 blueprint 数据")
            return
        print(f"收到运行蓝图请求: {blueprint}")

        # 两个回调把引擎的执行进度实时推送给前端
        async def onMessage(nodeId, result):
            await sendMessage(ws, "nodeResult", id, {"nodeId": nodeId, "result": result})

        async def onError(nodeId, error):
            await sendError(ws, "nodeError", id, {"nodeId": nodeId, "error": error})

        result = await engine.run(blueprint, onMessage, onError)
        await sendMessage(ws, "blueprintComplete", id, {"result": result})

    else:
        await sendError(ws, "unknown", id, f"未知消息类型：{msg_type}")


async def handleConnection(ws):
    """
    管理单个客户端连接的完整生命周期：加入 → 收消息 → 离开。
    """
    clients.add(ws)
    print(f"前端已连接，当前连接数: {len(clients)}")

    async for message in ws:  # 连接断开时自动停止迭代，无需捕获异常
        await handleMessage(ws, message)

    clients.discard(ws)  # 用 discard 而非 remove，防止重复移除时抛异常
    print(f"前端已断开，当前连接数: {len(clients)}")


def start(host="0.0.0.0", port=8765):
    """
    启动 WebSocket 服务器，阻塞运行直到进程终止。
    """
    print(f"WebSocket服务启动中... ws://{host}:{port}")

    async def main():
        async with websockets.serve(handleConnection, host, port):
            print(f"WebSocket服务已启动: ws://{host}:{port}")

            # 可拔插热重载：需要时保留下面两行，不需要时注释即可拔出
            import plugin_hot_reload

            plugin_hot_reload.mountHotReload(asyncio.get_running_loop(), broadcast)

            await asyncio.Future()  # 永久挂起，保持服务常驻

    asyncio.run(main())
