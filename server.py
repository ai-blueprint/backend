"""
server.py - WebSocket服务器

用法：
    import server
    server.start()              # 使用默认参数启动
    server.start("0.0.0.0", 9000)  # 指定host和port启动
"""

import asyncio
import json
import threading  # 线程事件能力，用于安全取消后台训练
import websockets

import registry  # 节点注册表，提供 getAllForFrontend()
import engine  # 蓝图执行引擎，提供 run()
import operations  # 跑分、训练、检查点和导出指令，不直接操作WebSocket
import plugins  # 本地可信插件指令，提供查询和重载能力

clients = set()  # 已连接的前端客户端集合，用 set 存储方便增删
trainingJobs = {}  # 按连接和请求ID保存训练任务，其他客户端不能取消
clientModels = {}  # 每个连接保存最近训练或加载的持久模型


async def sendMessage(ws, messageType, messageId, messageData):
    """
    向单个客户端发送普通消息。
    消息格式：{"type": ..., "id": ..., "data": ...}
    messageId 用于前端将响应匹配回对应的请求。
    """
    try:
        await ws.send(json.dumps({"type": messageType, "id": messageId, "data": messageData}, allow_nan=False))  # 严格JSON阻止特殊数值破坏浏览器解析
    except Exception as error:
        print(f"发送消息失败（{messageType}）: {error}")  # 客户端已断开，记录错误日志


async def sendError(ws, messageType, messageId, errorMsg):
    """
    向单个客户端发送错误消息。
    消息格式：{"type": ..., "id": ..., "error": ...}
    用 error 字段而非 data，方便前端区分正常响应和错误响应。
    """
    try:
        await ws.send(json.dumps({"type": messageType, "id": messageId, "error": errorMsg}, allow_nan=False))  # 错误信封同样使用严格JSON
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
            await ws.send(json.dumps(payload, allow_nan=False))  # 广播也必须保持严格JSON
        except Exception as error:
            print(f"广播消息失败: {error}")  # 客户端已断开，记录日志但继续发送给其他人


def getErrorData(error, code="requestFailed"):
    if isinstance(error, engine.BlueprintError):
        return error.toData()  # 业务错误保留稳定代码和具体上下文
    return {"code": code, "message": str(error), "details": {}}  # 未分类异常统一为结构化错误


def getRequestData(requestData):
    messageData = requestData.get("data")  # 所有业务请求都从固定 data 字段读取参数
    if not isinstance(messageData, dict):
        raise engine.BlueprintError("invalidMessage", "消息data必须是对象")  # 入口卫语句阻止业务指令处理畸形数据
    return messageData  # 返回经过类型校验的业务数据


def getBlueprint(messageData):
    blueprint = messageData.get("blueprint")  # 蓝图由各命令共享同一字段名称
    if not isinstance(blueprint, dict):
        raise engine.BlueprintError("missingBlueprint", "缺少有效的blueprint对象")  # 空值和数组都不能进入编译器
    return blueprint  # 返回显式图对象供指令执行


def getClientModel(ws, blueprint):
    session = clientModels.get(ws)  # 查询当前连接最近一次持久模型
    if not session or session.get("blueprint") != blueprint:
        return None  # 蓝图变化后旧权重不能套用到不同图结构
    return session.get("model")  # 返回与当前蓝图完全匹配的模型


async def runTraining(ws, messageId, messageData, cancelEvent):
    loop = asyncio.get_running_loop()  # 主事件循环负责真正发送WebSocket反馈

    def reportProgress(progressData):
        sendFuture = asyncio.run_coroutine_threadsafe(sendMessage(ws, "trainProgress", messageId, progressData), loop)  # 将发送动作安全提交回主事件循环
        sendFuture.result()  # 等待本条进度发出，保证终态不会越过最后一条进度

    try:
        blueprint = getBlueprint(messageData)  # 后台任务开始时读取已校验蓝图
        result = await asyncio.to_thread(operations.trainBlueprint, blueprint, messageData.get("training"), reportProgress, cancelEvent)  # CPU训练在线程中执行，不阻塞连接循环
        trainedModel = result.pop("model", None)  # nn.Module 只在后端内部流转，不能进入JSON协议
        if trainedModel is not None and ws in clients:
            clientModels[ws] = {"blueprint": blueprint, "model": trainedModel}  # 后续运行、保存和导出复用训练权重
        checkpointPath = messageData.get("checkpointPath")  # 可选完成后保存本次训练参数
        if checkpointPath and result.get("status") == "complete":
            checkpoint = await asyncio.to_thread(operations.saveCheckpoint, blueprint, checkpointPath, trainedModel, {"training": result})  # 保存同一持久模型的训练状态
            result["checkpoint"] = checkpoint  # 完成反馈附带受控检查点路径
        await sendMessage(ws, "trainComplete", messageId, result)  # 成功或取消都只发送一次训练终态
    except Exception as error:
        await sendError(ws, "trainError", messageId, getErrorData(error, "trainingFailed"))  # 异常只发送一次错误终态
    finally:
        trainingJobs.pop((ws, messageId), None)  # 训练终止后释放当前连接拥有的任务


async def handleMessage(ws, message):
    """
    触发事件入口：只解析消息、选择指令，并把指令结果反馈给当前客户端。
    前端消息格式：{"type": ..., "id": ..., "data": ...}
    """
    try:
        requestData = json.loads(message)  # 原始文本先转换为协议对象
    except (json.JSONDecodeError, TypeError) as error:
        await sendError(ws, "parseError", "", {"code": "invalidJSON", "message": f"无效的JSON格式: {error}", "details": {}})  # 解析失败直接结构化反馈
        return
    if not isinstance(requestData, dict):
        await sendError(ws, "parseError", "", {"code": "invalidMessage", "message": "消息必须是JSON对象", "details": {}})  # 非对象没有类型和请求ID
        return

    messageType = requestData.get("type", "")  # 类型决定触发哪个业务指令
    messageId = str(requestData.get("id", ""))  # 请求ID原样关联全部流式和终态反馈
    if not isinstance(messageType, str) or not messageType:
        await sendError(ws, "parseError", messageId, {"code": "missingType", "message": "缺少有效的消息type", "details": {}})  # 无类型无法路由
        return

    try:
        if messageType == "getRegistry":
            await sendMessage(ws, "getRegistry", messageId, registry.getAllForFrontend())  # 读取注册表并立即反馈
            return

        messageData = getRequestData(requestData)  # 其余业务消息都必须携带对象data
        if messageType == "runBlueprint":
            blueprint = getBlueprint(messageData)  # 读取本次预览图定义

            async def onNodeResult(nodeId, nodeResult):
                await sendMessage(ws, "nodeResult", messageId, {"nodeId": nodeId, **nodeResult})  # 全端口和耗时按节点流式反馈

            async def onNodeError(nodeId, nodeError):
                await sendError(ws, "nodeError", messageId, {"nodeId": nodeId, **nodeError})  # 节点错误保留稳定代码和上下文

            result = await engine.run(blueprint, onNodeResult, onNodeError, messageData.get("inputs"), messageData.get("maxValues", 256), getClientModel(ws, blueprint))  # 匹配时复用训练或加载权重
            await sendMessage(ws, "blueprintComplete", messageId, result)  # 无论成功失败都恰好发送一次蓝图终态
            return

        if messageType in {"scoreBlueprint", "score"}:
            blueprint = getBlueprint(messageData)  # 跑分和会话模型使用同一图身份
            result = await asyncio.to_thread(operations.scoreBlueprint, blueprint, messageData.get("score"), getClientModel(ws, blueprint))  # 跑分在线程中避免阻塞消息循环
            await sendMessage(ws, "scoreComplete", messageId, result)  # 返回延迟和参数数量
            return

        if messageType == "trainBlueprint":
            if not messageId:
                raise engine.BlueprintError("missingId", "训练请求必须提供id以支持进度和取消")  # 长任务必须有稳定关联ID
            jobKey = (ws, messageId)  # 连接和请求ID共同确定训练所有权
            if jobKey in trainingJobs:
                raise engine.BlueprintError("trainingExists", "相同id的训练任务已在运行")  # 防止重复任务覆盖取消句柄
            if any(owner is ws for owner, _ in trainingJobs):
                raise engine.BlueprintError("trainingLimit", "每个连接同时只能运行一个训练任务")  # 防止单客户端创建无界后台任务
            getBlueprint(messageData)  # 创建后台任务前同步拒绝缺失蓝图
            cancelEvent = threading.Event()  # 线程安全事件在批次边界触发取消
            trainingTask = asyncio.create_task(runTraining(ws, messageId, messageData, cancelEvent))  # 后台任务让当前连接继续接收取消请求
            trainingJobs[jobKey] = {"task": trainingTask, "cancel": cancelEvent}  # 保存带连接所有权的任务生命周期数据
            return

        if messageType == "loadCheckpoint":
            result = await asyncio.to_thread(operations.loadCheckpoint, messageData.get("path"))  # 加载并严格校验检查点内容
            loadedModel = result.pop("model", None)  # nn.Module 仅供后端后续计算，不能进入 JSON 响应
            clientModels[ws] = {"blueprint": result["blueprint"], "model": loadedModel}  # 后续运行和导出复用恢复权重
            await sendMessage(ws, "checkpointLoadComplete", messageId, result)  # 返回蓝图和清单供前端恢复工作区
            return

        if messageType == "cancelTraining":
            trainingId = str(messageData.get("trainingId", messageId))  # 默认取消与请求ID相同的训练
            job = trainingJobs.get((ws, trainingId))  # 只查询当前连接拥有的后台任务
            if not job:
                raise engine.BlueprintError("trainingNotFound", f"训练任务不存在: {trainingId}")  # 已结束任务无需取消
            job["cancel"].set()  # 通知训练线程在下一批次边界安全收口
            await sendMessage(ws, "cancelComplete", messageId, {"status": "requested", "trainingId": trainingId})  # 反馈取消请求已送达
            return

        if messageType == "saveCheckpoint":
            blueprint = getBlueprint(messageData)  # 保存当前图定义和匹配会话权重
            result = await asyncio.to_thread(operations.saveCheckpoint, blueprint, messageData.get("path"), getClientModel(ws, blueprint))
            await sendMessage(ws, "checkpointSaveComplete", messageId, result)  # 返回检查点路径和清单
            return

        if messageType in {"exportPython", "exportONNX"}:
            blueprint = getBlueprint(messageData)  # 导出当前图定义和匹配会话权重
            model = getClientModel(ws, blueprint)  # 训练或加载后的模型优先于新初始化模型
            if messageType == "exportPython":
                result = await asyncio.to_thread(operations.exportPython, blueprint, messageData.get("path", "exports/blueprint.py"), model)
            else:
                result = await asyncio.to_thread(operations.exportONNX, blueprint, messageData.get("path", "exports/blueprint.onnx"), messageData.get("export"), model)
            await sendMessage(ws, "exportComplete", messageId, result)  # 两种格式使用统一导出终态
            return

        commandMap = {
            "listPlugins": (plugins.discoverPlugins, "pluginList", lambda: ()),
            "pluginStatus": (plugins.discoverPlugins, "pluginStatus", lambda: ()),
            "reloadPlugins": (plugins.reloadPlugins, "pluginReloadComplete", lambda: ()),
        }  # 简单指令映射保持触发、执行和反馈关系一眼可追踪
        if messageType in commandMap:
            command, responseType, getCommandArguments = commandMap[messageType]  # 读取对应纯业务指令和反馈类型
            result = await asyncio.to_thread(command, *getCommandArguments())  # 只在命中命令后校验并读取它需要的数据
            await sendMessage(ws, responseType, messageId, result)  # 将指令数据变化反馈给请求方
            return

        await sendError(ws, "unknown", messageId, {"code": "unknownMessage", "message": f"未知消息类型: {messageType}", "details": {}})  # 未注册触发类型明确拒绝
    except Exception as error:
        if messageType == "runBlueprint":
            await sendMessage(ws, "blueprintComplete", messageId, {"status": "failed", "error": getErrorData(error), "outputNodeIds": [], "errorCount": 1, "durationMs": 0.0})  # 畸形运行请求也使用唯一蓝图终态收口
            return
        errorTypeMap = {"scoreBlueprint": "scoreError", "score": "scoreError", "trainBlueprint": "trainError", "cancelTraining": "cancelError", "saveCheckpoint": "checkpointError", "loadCheckpoint": "checkpointError", "exportPython": "exportError", "exportONNX": "exportError", "listPlugins": "pluginError", "pluginStatus": "pluginError", "reloadPlugins": "pluginError"}  # 每类操作使用稳定错误终态
        await sendError(ws, errorTypeMap.get(messageType, f"{messageType}Error"), messageId, getErrorData(error))  # 所有捕获异常返回统一结构


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
        ownedJobs = [job for (owner, _), job in trainingJobs.items() if owner is ws]  # 找出断开连接拥有的训练
        for job in ownedJobs:
            job["cancel"].set()  # 通知后台训练在下一批次边界停止
        clientModels.pop(ws, None)  # 连接结束后释放模型参数占用
        clients.discard(ws)  # 无论何种原因断开，都要清理这个连接的资源
        print(f"前端已断开，当前连接数: {len(clients)}")  # 记录断开日志


def start(host="127.0.0.1", port=8765):
    """
    启动 WebSocket 服务器，阻塞运行直到进程终止。
    """
    print(f"WebSocket服务启动中... ws://{host}:{port}")  # 启动前提示

    async def main():
        plugins.reloadPlugins()  # 服务接收请求前加载可信目录中的全部启用插件
        async with websockets.serve(handleConnection, host, port):
            print(f"WebSocket服务已启动: ws://{host}:{port}")  # 启动成功提示

            # 可拔插热重载：需要时保留下面两行，不需要时注释即可拔出
            import plugin_hot_reload  # 导入热重载模块

            plugin_hot_reload.mountHotReload(asyncio.get_running_loop(), broadcast)  # 挂载热重载机制

            await asyncio.Future()  # 永久挂起阻塞，保持服务器常驻

    asyncio.run(main())  # 启动异步事件循环
