"""
engine.py - 持久蓝图模型与图执行指令

BlueprintModel 在编译阶段把节点放入 ModuleDict，随后由同一组参数完成预览、训练、
跑分和导出。forward 接收按输入节点 ID 命名的值，返回按输出节点 ID 命名的值。
调用示例：model = compileBlueprint(blueprint); outputs = model({"input-1": tensor})
"""

import inspect  # 回调能力，用于兼容同步和异步通知
import math  # 有限数值判断能力，用于生成浏览器可解析JSON
import time  # 单调计时能力，用于记录节点和蓝图耗时
from collections import defaultdict  # 边分组能力，用于快速收集节点输入

import torch  # 张量能力，用于模型执行和结果序列化
import torch.nn as nn  # 神经网络容器，用于持久注册蓝图参数

import loader  # 节点加载能力，导入内置节点定义
import registry  # 节点注册能力，按操作码创建节点实例
import sort  # 图排序能力，确定稳定执行顺序


loader.loadAll()  # 进程启动时装载内置节点，编译请求可立即使用注册表


class BlueprintError(Exception):
    """携带稳定错误代码和上下文，供所有协议入口返回统一结构。"""

    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code = code  # 稳定代码供前端判断错误类别
        self.details = details or {}  # 结构化上下文供定位具体节点或字段

    def toData(self):
        return {"code": self.code, "message": str(self), "details": self.details}  # 输出协议可直接序列化的数据


class BlueprintModel(nn.Module):
    """按拓扑顺序路由端口，并通过 ModuleDict 持久拥有全部节点参数。"""

    # --- 编译蓝图图结构 ---
    def __init__(self, blueprint):
        super().__init__()
        if not isinstance(blueprint, dict):
            raise BlueprintError("invalidBlueprint", "blueprint必须是对象")  # 图结构不是对象时无法继续编译

        self.blueprint = blueprint  # 保留原始定义供检查点和导出落盘
        self.nodes = blueprint.get("nodes", [])  # 节点数据决定模型包含哪些模块
        self.edges = blueprint.get("edges", [])  # 边数据决定端口之间如何传值
        self.sortedIDs = sort.topoSort(self.nodes, self.edges, strict=True)  # 严格排序尽早拒绝悬空边
        self.nodeData = {node.get("id", ""): node for node in self.nodes}  # 建立节点 ID 到配置的显式索引
        self.edgesByTarget = defaultdict(list)  # 按目标节点保存边，执行时只读取相关输入
        for edge in self.edges:
            self.edgesByTarget[edge.get("target", "")].append(edge)  # 保留蓝图边顺序，端口覆盖行为可预测

        nodeModules = {}  # ModuleDict 需要不含点号的稳定内部键
        self.moduleKeys = {}  # 前端节点 ID 映射到安全内部键，图协议仍保留原 ID
        for nodeIndex, nodeID in enumerate(self.sortedIDs):
            node = self.nodeData[nodeID]  # 拓扑 ID 已由严格排序验证存在
            moduleKey = f"node_{nodeIndex}"  # 顺序键避开 ModuleDict 禁止的点号等前端字符
            self.moduleKeys[nodeID] = moduleKey  # 路由时通过映射找到持久模块
            nodeModules[moduleKey] = self._createNode(nodeID, node)  # 创建一次，后续运行复用同一参数
        self.nodeModules = nn.ModuleDict(nodeModules)  # 将所有可学习参数注册进 PyTorch 模型树

        self.inputIDs = [nodeID for nodeID in self.sortedIDs if self._getOpcode(nodeID) == "input"]  # 显式输入由输入节点 ID 标识
        self.outputIDs = [nodeID for nodeID in self.sortedIDs if self._getOpcode(nodeID) == "output"]  # 显式输出由输出节点 ID 标识
        if not self.outputIDs:
            sourceIDs = {edge.get("source", "") for edge in self.edges}  # 找出仍向下游供值的节点
            self.outputIDs = [nodeID for nodeID in self.sortedIDs if nodeID not in sourceIDs]  # 无输出节点时使用叶节点兼容旧蓝图

        self.lastNodeResults = {}  # 保存最近一次完整端口结果，预览回调和调试可读取
        self.lastNodeDurations = {}  # 保存最近一次逐节点耗时，结构化反馈统一消费

    # --- 校验并创建单个节点 ---
    def _createNode(self, nodeID, node):
        data = node.get("data", {})  # 前端把节点类型和参数存放在 data 字段
        opcode = data.get("opcode", "")  # 操作码用于查找注册定义
        if opcode not in registry.nodes:
            raise BlueprintError("unknownNode", f"未知的节点类型: {opcode}", {"nodeId": nodeID, "opcode": opcode})  # 未注册类型无法编译
        try:
            return registry.createNode(opcode, nodeID, data.get("params", {}))  # 参数只在编译时校验并创建模块
        except Exception as error:
            raise BlueprintError("nodeBuildFailed", f"创建节点实例失败: {error}", {"nodeId": nodeID, "opcode": opcode}) from error

    # --- 读取节点操作码 ---
    def _getOpcode(self, nodeID):
        return self.nodeData[nodeID].get("data", {}).get("opcode", "")  # 从原始蓝图读取稳定操作码

    # --- 按图路由并执行所有节点 ---
    def forward(self, modelInputs=None, nodeCallback=None):
        modelInputs = modelInputs or {}  # 空输入保留 InputNode 随机预览行为
        if not isinstance(modelInputs, dict):
            raise BlueprintError("invalidInputs", "inputs必须是按输入节点ID命名的对象")  # 显式映射避免多输入顺序歧义

        nodeResults = {}  # 当前 forward 内保存全部端口值供后续边读取
        nodeDurations = {}  # 当前 forward 内记录逐节点计算时间
        for nodeID in self.sortedIDs:
            inputValues = self._collectInputs(nodeID, nodeResults, modelInputs)  # 先沿边和模型输入收集命令数据
            startedAt = time.perf_counter()  # 节点执行前记录单调时钟
            try:
                outputValues = self.nodeModules[self.moduleKeys[nodeID]](inputValues)  # 用安全内部键调用持久节点模块
            except Exception as error:
                raise BlueprintError("nodeExecutionFailed", f"节点执行失败: {error}", {"nodeId": nodeID, "opcode": self._getOpcode(nodeID)}) from error
            if not isinstance(outputValues, dict):
                raise BlueprintError("invalidNodeOutput", "节点输出必须是端口对象", {"nodeId": nodeID})  # 图路由依赖命名端口
            nodeDurations[nodeID] = (time.perf_counter() - startedAt) * 1000  # 毫秒便于协议直接展示
            nodeResults[nodeID] = outputValues  # 完整端口值进入图数据流
            if nodeCallback:
                nodeCallback(nodeID, outputValues, nodeDurations[nodeID])  # 计算后反馈节点结果，不改变图数据

        self.lastNodeResults = nodeResults  # 一次执行完成后原子替换最近结果
        self.lastNodeDurations = nodeDurations  # 一次执行完成后原子替换耗时记录
        return {nodeID: self._getOutputValue(nodeID, nodeResults[nodeID]) for nodeID in self.outputIDs}  # 返回显式模型输出

    # --- 合并边输入和显式模型输入 ---
    def _collectInputs(self, nodeID, nodeResults, modelInputs):
        inputValues = {}  # 每个目标端口最多保留一个来源值
        for edge in self.edgesByTarget.get(nodeID, []):
            sourceID = edge.get("source", "")  # 来源节点由严格拓扑校验保证存在
            sourcePort = edge.get("sourceHandle", "out")  # 缺省输出端口沿用旧协议
            targetPort = edge.get("targetHandle", "in")  # 缺省输入端口沿用旧协议
            if sourcePort not in nodeResults.get(sourceID, {}):
                raise BlueprintError("missingOutputPort", f"来源端口不存在: {sourceID}.{sourcePort}", {"nodeId": nodeID})  # 静默传 None 会掩盖接线错误
            inputValues[targetPort] = nodeResults[sourceID][sourcePort]  # 将来源端口值写入目标端口
        if self._getOpcode(nodeID) == "input" and nodeID in modelInputs:
            inputValues["value"] = modelInputs[nodeID]  # 显式输入覆盖随机预览生成
        return inputValues  # 节点只看到属于自己的命名输入

    # --- 统一叶节点和 OutputNode 的输出值 ---
    def _getOutputValue(self, nodeID, outputValues):
        if self._getOpcode(nodeID) == "output":
            return outputValues.get("out")  # OutputNode 以 out 保存接收到的终点值
        if "out" in outputValues:
            return outputValues["out"]  # 普通单输出叶节点保持直观返回值
        return outputValues  # 多端口叶节点保留全部端口


# --- 编译持久蓝图模型 ---
def compileBlueprint(blueprint):
    with registry.registryLock:
        return BlueprintModel(blueprint)  # 编译期间阻止插件重载替换注册项


# --- 将 JSON 输入规格转换为张量 ---
def parseInputs(rawInputs):
    if rawInputs is None:
        return {}  # 未提供输入时由 InputNode 生成随机预览数据
    if not isinstance(rawInputs, dict):
        raise BlueprintError("invalidInputs", "inputs必须是对象")  # 多输入必须按节点 ID 明确命名
    return {nodeID: parseTensor(value) for nodeID, value in rawInputs.items()}  # 每个输入独立转换并保留节点名称


# --- 校验输入名称属于当前模型 ---
def validateModelInputs(model, modelInputs):
    unknownIDs = sorted(set(modelInputs) - set(model.inputIDs))  # 找出无法被任何InputNode消费的键
    if unknownIDs:
        raise BlueprintError("unknownInput", "inputs包含未知输入节点", {"inputIds": unknownIDs})  # 防止拼写错误静默回退随机输入
    return modelInputs  # 返回原映射便于调用链线性组合


# --- 读取输入节点已校验形状 ---
def getInputShape(model, inputID):
    module = model.nodeModules[model.moduleKeys[inputID]]  # 编译节点已把前端参数对象解包为业务值
    shape = module.params.get("out_shape", [4, 4])  # 输入节点缺省沿用预览形状
    if not isinstance(shape, (list, tuple)) or not shape or not all(isinstance(size, int) and size > 0 for size in shape):
        raise BlueprintError("invalidInputShape", f"输入节点形状无效: {inputID}", {"nodeId": inputID, "shape": shape})  # torch.randn需要正整数序列
    return list(shape)  # 返回独立列表避免修改节点参数


# --- 转换单个JSON安全数值 ---
def serializeNumber(value):
    if isinstance(value, complex):
        return {"real": serializeNumber(value.real), "imag": serializeNumber(value.imag)}  # 复数显式拆分实部和虚部
    if isinstance(value, float) and not math.isfinite(value):
        return None  # JSON没有NaN和无穷值，使用null保持协议合法
    return value  # 有限整数、浮点和布尔值可直接编码


# --- 转换单个输入值 ---
def parseTensor(value):
    if isinstance(value, torch.Tensor):
        return value  # 内部训练和测试可直接传入张量，不做无意义复制
    if isinstance(value, dict) and "values" in value:
        dtypeName = value.get("dtype", "float32")  # 协议默认使用训练友好的 float32
        dtype = getattr(torch, dtypeName, None)  # 从公开 torch dtype 名称查找类型
        if not isinstance(dtype, torch.dtype):
            raise BlueprintError("invalidDtype", f"不支持的张量类型: {dtypeName}")  # 禁止任意属性被当作 dtype
        tensor = torch.tensor(value["values"], dtype=dtype)  # 将有界 JSON 数值构造成张量
        shape = value.get("shape")  # 可选形状用于扁平 values 的明确还原
        return tensor.reshape(shape) if shape is not None else tensor  # 有形状时验证元素数量并重排
    if isinstance(value, (list, int, float, bool)):
        return torch.tensor(value)  # 简写输入使用 PyTorch 的自然类型推断
    raise BlueprintError("invalidTensor", "输入值必须是数组、数值或包含values的张量对象")  # 其他 JSON 类型不能形成模型输入


# --- 有界序列化张量和普通值 ---
def serializeValue(value, maxValues=256):
    maxValues = max(1, min(int(maxValues), 4096))  # 限制预览大小，避免单条消息耗尽内存
    if isinstance(value, torch.Tensor):
        detached = value.detach()  # 反馈数据不应继续持有反向图
        flatValues = detached.cpu().reshape(-1)  # 移到 CPU 后按一维截取固定数量
        shownValues = [serializeNumber(item) for item in flatValues[:maxValues].tolist()]  # 截断后替换JSON不支持的特殊数值
        return {
            "kind": "tensor",  # 前端按稳定类型选择标量、矩阵或高维预览
            "shape": list(detached.shape),  # 原始形状帮助前端理解完整张量
            "dtype": str(detached.dtype).removeprefix("torch."),  # 返回可用于输入协议的 dtype 名称
            "device": str(detached.device),  # 返回执行设备便于性能诊断
            "values": shownValues,  # 始终使用扁平有界值，避免嵌套大数组
            "truncated": detached.numel() > maxValues,  # 明确提示 values 是否只是一部分
            "totalElements": detached.numel(),  # 完整元素数量帮助前端解释截断比例
        }
    if isinstance(value, dict):
        return {str(key): serializeValue(item, maxValues) for key, item in value.items()}  # 多端口和状态对象逐字段序列化
    if isinstance(value, (list, tuple)):
        shownItems = list(value[:maxValues])  # 普通序列同样受上限保护
        return {"values": [serializeValue(item, maxValues) for item in shownItems], "truncated": len(value) > maxValues}  # 结构化描述截断状态
    if value is None or isinstance(value, (int, float, bool, complex)):
        return serializeNumber(value)  # 数值统一保证严格JSON可编码
    if isinstance(value, str):
        return value  # 字符串可直接返回
    return {"type": type(value).__name__, "value": str(value)[:512], "truncated": len(str(value)) > 512}  # 未知对象仅提供有界文本说明


# --- 异步运行预览并反馈节点结果 ---
async def run(blueprint, onMessage, onError, inputs=None, maxValues=256, compiledModel=None):
    startedAt = time.perf_counter()  # 蓝图总耗时从编译前开始计算
    try:
        model = compiledModel or compileBlueprint(blueprint)  # 会话模型保留训练或加载权重，普通预览按需编译
        model.eval()  # 预览默认关闭 dropout 等训练随机行为
        pendingReports = []  # 同步 forward 先收集结果，再按异步回调顺序发送

        def collectResult(nodeID, outputValues, durationMs):
            pendingReports.append((nodeID, outputValues, durationMs))  # 保留全部输出端口和节点耗时

        with torch.no_grad():
            modelInputs = validateModelInputs(model, parseInputs(inputs))  # 输入名称必须对应显式InputNode
            modelOutputs = model(modelInputs, collectResult)  # 无梯度预览降低内存占用
        for nodeID, outputValues, durationMs in pendingReports:
            opcode = model.nodeData[nodeID].get("data", {}).get("opcode", "")  # 节点类型随结果返回，前端无需反查易变化的注册表
            report = {"opcode": opcode, "outputs": serializeValue(outputValues, maxValues), "durationMs": durationMs}  # 每个节点返回所有端口
            callbackResult = onMessage(nodeID, report)  # 触发节点完成反馈
            if inspect.isawaitable(callbackResult):
                await callbackResult  # 异步 WebSocket 回调按拓扑顺序完成
        durationMs = (time.perf_counter() - startedAt) * 1000  # 编译和执行都计入用户感知耗时
        return {"status": "succeeded", "outputs": serializeValue(modelOutputs, maxValues), "outputNodeIds": model.outputIDs, "errorCount": 0, "durationMs": durationMs}  # 唯一成功终态数据
    except Exception as error:
        blueprintError = error if isinstance(error, BlueprintError) else BlueprintError("blueprintFailed", str(error))  # 未分类异常统一收口
        callbackResult = onError(blueprintError.details.get("nodeId", ""), blueprintError.toData())  # 流式反馈结构化错误
        if inspect.isawaitable(callbackResult):
            await callbackResult  # 确保错误先于终态返回
        durationMs = (time.perf_counter() - startedAt) * 1000  # 失败同样报告已消耗时间
        return {"status": "failed", "error": blueprintError.toData(), "outputNodeIds": [], "errorCount": 1, "durationMs": durationMs}  # 由调用方仍发送一次 blueprintComplete
