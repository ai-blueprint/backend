"""
server.py - WebSocket服务器

用法：
    import server
    server.start()  # 使用默认参数启动
    server.start("0.0.0.0", 9000)  # 指定host和port启动

示例：
    server.start()  # 在0.0.0.0:8765启动WebSocket服务
    server.start("0.0.0.0")  # 在0.0.0.0:8765启动
    server.start(port=9000)  # 在0.0.0.0:9000启动
"""

import asyncio  # 异步IO库，用于处理WebSocket的异步操作
import json  # JSON库，用于消息的序列化和反序列化
import os  # 操作系统模块，用于路径操作
import websockets  # WebSocket库，用于创建WebSocket服务器

import registry  # 节点注册表模块，用于获取节点信息
import engine  # 蓝图执行引擎模块，用于运行蓝图
import loader  # 节点加载器模块，用于热重载

clients = set()  # 全局变量：已连接的前端客户端集合，用set存储方便增删


async def sendMessage(ws, type, id, data):
    """
    发送消息给前端
    示例：
        await sendMessage(websocket, "getNodes", "req1", nodesData)  # 发送节点数据
        await sendMessage(websocket, "nodeComplete", "req2", result)  # 发送节点执行结果
    """
    msg = {}  # 创建空字典准备装消息
    msg["type"] = type  # 消息类型，比如getNodes、runBlueprint
    msg["id"] = id  # 消息ID，用于前端匹配请求和响应
    msg["data"] = data  # 消息数据，具体内容根据type不同而不同
    print(f"发送给前端消息: {type} {data}")
    text = json.dumps(msg)  # 把字典转成JSON字符串

    await ws.send(text)  # 通过WebSocket发送给前端


async def sendError(ws, type, id, error):
    """
    发送错误消息给前端
    用法：await sendError(ws, "runBlueprint", "msg123", "节点执行失败")
    """
    msg = {}  # 创建空字典准备装错误消息
    msg["type"] = type  # 消息类型
    msg["id"] = id  # 消息ID
    msg["error"] = error  # 错误信息
    text = json.dumps(msg)  # 把字典转成JSON字符串
    await ws.send(text)  # 通过WebSocket发送给前端


async def handleMessage(ws, message):
    """
    用法：await handleMessage(ws, '{"type": "getNodes", "id": "req1"}')
    """
    data = json.loads(message)  # 把JSON字符串解析成字典
    msg_type = data.get("type", "")  # 提取消息类型，默认空字符串
    id = data.get("id", "")  # 提取消息ID，默认空字符串

    if msg_type == "getRegistry":  # 如果是请求节点注册表
        result = registry.getAllForFrontend()  # 调用registry获取前端格式的节点数据
        await sendMessage(ws, msg_type, id, result)  # 发送响应给前端
        return  # 处理完毕，返回

    elif msg_type == "runBlueprint":  # 如果是请求运行蓝图
        blueprint = data["data"].get("blueprint")  # 提取蓝图数据
        print(f"收到运行蓝图请求: {blueprint}")  # 收到运行蓝图请求: None

        # 前置校验：对每个节点的params做range/type修正
        for node in blueprint.get("nodes", []):  # 遍历蓝图中所有节点
            nodeData = node.get("data", {})  # 获取节点数据
            opcode = nodeData.get("opcode", "")  # 获取节点opcode
            params = nodeData.get("params", {})  # 获取节点参数
            if opcode in registry.nodes:  # 仅校验已注册的节点
                nodeData["params"] = registry.validateParams(opcode, params)  # 校验并修正参数，写回蓝图

        async def onMessage(nodeId, result):  # 定义节点执行完成的回调
            await sendMessage(ws, "nodeResult", id, {"nodeId": nodeId, "result": result})  # 发送节点结果

        async def onError(nodeId, error):  # 定义节点执行出错的回调
            await sendError(ws, "nodeError", id, {"nodeId": nodeId, "error": error})  # 发送节点错误

        result = await engine.run(blueprint, onMessage, onError)  # 调用引擎运行蓝图
        await sendMessage(ws, "blueprintComplete", id, {"result": result})  # 发送蓝图执行完成消息
        return  # 处理完毕，返回

    else:  # 如果是未知消息类型
        await sendError(ws, "unknown", id, f"未知消息类型：{msg_type}")  # 发送未知消息类型的错误消息
        return


async def handleConnection(ws):
    clients.add(ws)  # 将新连接的前端加入clients集合
    print(f"前端已连接，当前连接数: {len(clients)}")  # 打印连接信息

    try:  # 尝试接收消息
        async for message in ws:  # 循环接收前端发来的消息
            await handleMessage(ws, message)  # 调用handleMessage处理每条消息
    except websockets.exceptions.ConnectionClosed:  # 如果连接断开
        pass  # 忽略断开异常，正常退出循环
    finally:  # 无论如何都要执行的清理
        clients.discard(ws)  # 从clients集合中移除这个连接
        print(f"前端已断开，当前连接数: {len(clients)}")  # 打印断开信息


def start(host="0.0.0.0", port=8765):
    print(f"WebSocket服务启动中... ws://{host}:{port}")  # 打印启动信息

    async def broadcast(type, data):  # 广播消息给所有已连接客户端
        msg = json.dumps({"type": type, "data": data})  # 序列化消息
        for ws in list(clients):  # 遍历所有客户端
            try:
                await ws.send(msg)  # 发送消息
            except Exception:
                pass  # 忽略发送失败的客户端

    async def watchNodes():  # 监控nodes目录文件变化
        from watchfiles import awatch  # 延迟导入watchfiles
        nodesDir = os.path.join(os.path.dirname(__file__), "nodes")  # 获取nodes目录绝对路径
        print(f"开始监控节点目录: {nodesDir}")  # 打印监控信息
        async for changes in awatch(nodesDir):  # 异步监控文件变化
            pyChanges = [c for c in changes if c[1].endswith(".py")]  # 只关注.py文件变化
            if not pyChanges:  # 没有.py文件变化
                continue  # 跳过
            print(f"检测到节点文件变化: {pyChanges}")  # 打印变化信息
            snapshot = dict(registry.nodes), dict(registry.categories)  # 快照当前registry
            try:
                loader.reloadAll()  # 全量重建registry
                print("热重载完成，广播新registry")  # 打印成功信息
                await broadcast("registryUpdated", registry.getAllForFrontend())  # 广播给前端
            except Exception as e:  # 重载失败
                print(f"热重载失败，回滚: {e}")  # 打印失败信息
                registry.nodes.clear()  # 清空当前
                registry.categories.clear()  # 清空当前
                registry.nodes.update(snapshot[0])  # 回滚nodes
                registry.categories.update(snapshot[1])  # 回滚categories
                await broadcast("reloadError", {"error": str(e)})  # 广播错误消息

    async def main():  # 定义异步主函数
        async with websockets.serve(handleConnection, host, port):  # 创建WebSocket服务器
            print(f"WebSocket服务已启动: ws://{host}:{port}")  # 打印启动成功信息
            await watchNodes()  # 监控nodes目录变化，替代原来的await asyncio.Future()

    asyncio.run(main())  # 运行异步主函数
