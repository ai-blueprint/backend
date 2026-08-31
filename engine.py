"""
engine.py - 持久蓝图模型与图执行指令

BlueprintModel 在编译阶段把输入可达节点放入 ModuleDict，随后逐节点传播随机张量。
forward 接收按输入节点 ID 命名的值，返回按输出节点 ID 命名的值。
调用示例：model = compileBlueprint(blueprint); outputs = model({"input-1": tensor})
"""

import inspect  # 回调能力，用于兼容同步和异步通知
import json  # 缓存键编码能力，用于比较蓝图业务内容是否变化
import math  # 有限数值判断能力，用于生成浏览器可解析JSON
import zlib  # 稳定哈希能力，用于把节点ID转换为固定初始化种子
import time  # 单调计时能力，用于记录节点和蓝图耗时
from collections import defaultdict  # 边分组能力，用于快速收集节点输入

import torch  # 张量能力，用于模型执行和结果序列化
import torch.nn as nn  # 神经网络容器，用于持久注册蓝图参数

import expressions  # 变量表达式能力，把参数中的变量引用消解为具体数值
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


# --- 计算节点的固定初始化种子 ---
def getNodeSeed(nodeID):
    """
    用法：seed = getNodeSeed("linear-1")  # 同一节点ID永远得到同一初始化种子
    """
    return zlib.crc32(str(nodeID).encode("utf-8"))  # CRC32稳定且跨进程一致，节点ID不变则权重初始化不变


# --- 筛选所有能接收到输入传播的节点 ---
def selectActiveGraph(nodes, edges):
    sort.topoSort(nodes, [], strict=True)  # 校验重复节点ID，但不让断开区域的循环影响实验路径
    nodeData = {node.get("id", ""): node for node in nodes}  # 建立完整节点索引用于校验边和读取类型
    for edge in edges:
        sourceID = edge.get("source", "")  # 读取边的来源节点
        targetID = edge.get("target", "")  # 读取边的目标节点
        if sourceID not in nodeData:
            raise BlueprintError("invalidEdge", f"源节点不存在: {sourceID}", {"source": sourceID, "target": targetID})  # 悬空来源不能进入路径分析
        if targetID not in nodeData:
            raise BlueprintError("invalidEdge", f"目标节点不存在: {targetID}", {"source": sourceID, "target": targetID})  # 悬空目标不能进入路径分析

    inputIDs = {nodeID for nodeID, node in nodeData.items() if node.get("data", {}).get("opcode") == "input"}  # 输入节点是传播起点
    if not inputIDs:
        raise BlueprintError("missingExperimentPath", "蓝图必须包含Input节点")  # 实时传播必须有随机张量起点

    outgoing = defaultdict(list)  # 保存每个节点可继续传播到的下游
    for edge in edges:
        outgoing[edge.get("source", "")].append(edge.get("target", ""))  # 建立正向邻接关系

    reachableFromInput = set(inputIDs)  # 从所有输入节点开始向前搜索
    pendingIDs = list(inputIDs)  # 使用显式队列保持数据流易追踪
    for nodeID in pendingIDs:
        for targetID in outgoing.get(nodeID, []):
            if targetID in reachableFromInput:
                continue  # 已访问节点无需重复加入队列
            reachableFromInput.add(targetID)  # 标记节点确实接收到输入传播
            pendingIDs.append(targetID)  # 继续搜索它的下游

    activeIDs = reachableFromInput  # 输入自身及其全部下游都参与传播，不要求分支最终连接Output
    activeNodes = [node for node in nodes if node.get("id", "") in activeIDs]  # 保留原蓝图节点顺序
    activeEdges = [edge for edge in edges if edge.get("source", "") in activeIDs and edge.get("target", "") in activeIDs]  # 只保留输入可达区域内部连线
    return activeNodes, activeEdges  # 编译器后续看不到没有输入来源的节点


class BlueprintModel(nn.Module):
    """按拓扑顺序路由端口，并通过 ModuleDict 持久拥有全部节点参数。"""

    # --- 编译蓝图图结构 ---
    def __init__(self, blueprint):
        super().__init__()
        if not isinstance(blueprint, dict):
            raise BlueprintError("invalidBlueprint", "blueprint必须是对象")  # 图结构不是对象时无法继续编译

        self.blueprint = blueprint  # 保留原始定义供实验信息追踪
        try:
            self.variables = expressions.getVariablesMap(blueprint.get("variables", []))  # 蓝图级变量表供所有节点参数引用
        except ValueError as error:
            raise BlueprintError("invalidVariable", str(error)) from error  # 变量定义错误统一走结构化协议
        allNodes = blueprint.get("nodes", [])  # 读取画布全部节点，其中可能包含断开的教学草稿
        allEdges = blueprint.get("edges", [])  # 读取画布全部连线用于输入可达性分析
        self.nodes, self.edges = selectActiveGraph(allNodes, allEdges)  # 裁剪为输入节点能够传播到的全部下游
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

        self.inputIDs = [nodeID for nodeID in self.sortedIDs if self._getOpcode(nodeID) == "input"]  # 包含所有随机传播起点
        self.outputIDs = [nodeID for nodeID in self.sortedIDs if self._getOpcode(nodeID) == "output"]  # 只包含真正接到输入的输出节点
        self.targetNodeIDs = self._findTargetBranchNodes()  # 目标分支只生成答案，不参与权重更新

        self.lastNodeResults = {}  # 保存最近一次完整端口结果，预览回调和调试可读取
        self.lastNodeDurations = {}  # 保存最近一次逐节点耗时，结构化反馈统一消费
        self.trainingInputs = {}  # 保存训练批次，连续训练时每一步使用同一组输入和目标
        trainableParameters = [parameter for nodeID in self.sortedIDs if nodeID not in self.targetNodeIDs for parameter in self.nodeModules[self.moduleKeys[nodeID]].parameters() if parameter.requires_grad]  # 只收集预测分支的可学习参数
        self.optimizerConfig = {"name": "sgd", "learningRate": 0.01, "gradientClip": 0.0}  # 保存当前训练超参数，蓝图变化时重新使用默认值
        self.optimizer = torch.optim.SGD(trainableParameters, lr=0.01) if trainableParameters else None  # 无可学习参数的普通蓝图不创建空优化器

    # --- 校验并创建单个节点 ---
    def _createNode(self, nodeID, node):
        data = node.get("data", {})  # 前端把节点类型和参数存放在 data 字段
        opcode = data.get("opcode", "")  # 操作码用于查找注册定义
        if not registry.hasNode(opcode):
            raise BlueprintError("unknownNode", f"未知的节点类型: {opcode}", {"nodeId": nodeID, "opcode": opcode})  # 未注册类型无法编译
        try:
            resolvedParams = expressions.resolveNodeParams(data.get("params", {}), self.variables)  # 变量引用在编译时消解为具体数值
        except ValueError as error:
            raise BlueprintError("invalidExpression", f"参数表达式无效: {error}", {"nodeId": nodeID, "opcode": opcode}) from error
        try:
            with torch.random.fork_rng(devices=[]):  # 隔离全局随机状态，节点初始化种子不影响随机输入生成
                torch.manual_seed(getNodeSeed(nodeID))  # 同一节点ID每次编译都用相同种子，权重跨轮次和重启保持一致
                return registry.createNode(opcode, nodeID, resolvedParams)  # 参数只在编译时校验并创建模块
        except Exception as error:
            raise BlueprintError("nodeBuildFailed", f"创建节点实例失败: {error}", {"nodeId": nodeID, "opcode": opcode}) from error

    # --- 读取节点操作码 ---
    def _getOpcode(self, nodeID):
        return self.nodeData[nodeID].get("data", {}).get("opcode", "")  # 从原始蓝图读取稳定操作码

    # --- 找出输出目标端口上游的目标分支 ---
    def _findTargetBranchNodes(self):
        incoming = defaultdict(list)  # 保存每个节点各端口的来源节点
        for edge in self.edges:
            incoming[(edge.get("target", ""), edge.get("targetHandle", "in"))].append(edge.get("source", ""))  # 按目标端口记录上游
        targetIDs = set()  # 保存所有目标分支节点
        pending = [sourceID for outputID in self.outputIDs for sourceID in incoming.get((outputID, "target"), [])]  # 从Output.target入口开始回溯
        while pending:
            nodeID = pending.pop()  # 取出一个待回溯节点
            if nodeID in targetIDs: continue  # 已访问节点无需重复回溯
            targetIDs.add(nodeID)  # 标记目标分支节点
            pending.extend(sourceID for (targetID, _), sourceIDs in incoming.items() if targetID == nodeID for sourceID in sourceIDs)  # 继续寻找更上游节点
        return targetIDs  # 返回目标分支节点集合

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

    # --- 执行一次带目标值的训练步骤 ---
    def configureOptimizer(self, optimizerName="sgd", learningRate=0.01):
        optimizerName = str(optimizerName).lower()  # 统一优化器名称大小写
        learningRate = float(learningRate)  # 统一学习率类型
        if optimizerName not in {"sgd", "adam"} or not math.isfinite(learningRate) or learningRate <= 0:
            raise BlueprintError("invalidTrainingConfig", "优化器只能选择SGD或Adam，学习率必须是正数")  # 拒绝无效训练配置
        if self.optimizerConfig["name"] == optimizerName and self.optimizerConfig["learningRate"] == learningRate:
            return  # 配置没有变化时保留已有优化器动量
        parameters = [parameter for nodeID in self.sortedIDs if nodeID not in self.targetNodeIDs for parameter in self.nodeModules[self.moduleKeys[nodeID]].parameters() if parameter.requires_grad]  # 只让预测分支参数进入优化器
        optimizerClass = torch.optim.Adam if optimizerName == "adam" else torch.optim.SGD  # 按用户选择创建优化器
        self.optimizer = optimizerClass(parameters, lr=learningRate)  # 更换优化器并保留当前模型权重
        self.optimizerConfig.update({"name": optimizerName, "learningRate": learningRate})  # 记录本次生效配置

    # --- 执行一次带目标值的训练步骤 ---
    def trainStep(self, maxValues=65536, optimizerName="sgd", learningRate=0.01, gradientClip=0.0):
        if self.optimizer is None:
            raise BlueprintError("noTrainableParameters", "蓝图中没有可训练参数，请加入Linear等可学习节点")  # 没有参数时无法执行权重更新
        self.configureOptimizer(optimizerName, learningRate)  # 应用用户本轮选择的优化器和学习率
        gradientClip = float(gradientClip)  # 统一梯度裁剪阈值类型
        if not math.isfinite(gradientClip) or gradientClip < 0:
            raise BlueprintError("invalidTrainingConfig", "梯度裁剪必须是大于等于0的数字")  # 零表示不裁剪
        self.train()  # 打开可学习节点的训练行为
        self.optimizer.zero_grad()  # 清除上一轮梯度，避免梯度跨轮累积
        if not self.trainingInputs:
            for inputID in self.inputIDs:
                self.trainingInputs[inputID] = self.nodeModules[self.moduleKeys[inputID]]({}).get("out").detach()  # 第一步生成固定训练批次，后续重复使用
        nodeResults = {}  # 保存本轮所有端口值供后续节点和损失读取
        for nodeID in self.sortedIDs:
            inputValues = self._collectInputs(nodeID, nodeResults, self.trainingInputs)  # 沿蓝图连线收集输入并复用固定训练批次
            if self._getOpcode(nodeID) != "input" and any(value is None for value in inputValues.values()):
                continue  # 目标或预测分支断路时跳过当前节点
            try:
                outputValues = self.nodeModules[self.moduleKeys[nodeID]](inputValues)  # 保留梯度执行真实节点
            except Exception as error:
                raise BlueprintError("nodeExecutionFailed", f"节点执行失败: {error}", {"nodeId": nodeID, "opcode": self._getOpcode(nodeID)}) from error
            if not isinstance(outputValues, dict):
                raise BlueprintError("invalidNodeOutput", "节点输出必须是端口对象", {"nodeId": nodeID})  # 图路由依赖命名端口
            nodeResults[nodeID] = outputValues  # 当前结果进入后续节点

        outputValues = [nodeResults[nodeID] for nodeID in self.outputIDs if nodeID in nodeResults]  # 只读取实际到达的输出节点
        trainingOutput = next((value for value in outputValues if value.get("target") is not None), None)  # 有目标端口才进入训练模式
        if not trainingOutput or trainingOutput.get("out") is None:
            raise BlueprintError("trainingNotConfigured", "请把预测分支和目标分支同时连接到输出节点")  # 没有完整训练接口时拒绝更新权重
        prediction = trainingOutput["out"]  # 输出节点的out代表模型预测结果
        target = trainingOutput["target"]  # 输出节点的target代表目标分支结果
        if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise BlueprintError("invalidTrainingData", "预测值和目标值必须是张量")  # 损失函数不能处理空值或普通对象
        target = target.detach()  # 目标分支只提供答案，不允许目标生成网络参与反向传播
        if prediction.shape != target.shape:
            raise BlueprintError("trainingShapeMismatch", f"预测形状{list(prediction.shape)}与目标形状{list(target.shape)}不一致", {"predictionShape": list(prediction.shape), "targetShape": list(target.shape)})  # 形状不同无法逐元素比较
        loss = torch.mean((prediction - target.to(prediction.dtype)) ** 2)  # 第一版使用通用均方误差，不引入额外训练节点
        if not torch.isfinite(loss):
            raise BlueprintError("invalidLoss", "损失值不是有限数字")  # 无效损失不能继续更新权重
        beforeWeights = {nodeID: {name: parameter.detach().clone() for name, parameter in self.nodeModules[self.moduleKeys[nodeID]].named_parameters()} for nodeID in self.sortedIDs}  # 训练前按参数名保存小型权重快照
        loss.backward()  # 计算所有预测分支可学习节点的梯度
        if gradientClip > 0:
            torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], gradientClip)  # 超过阈值时限制梯度，避免训练爆炸
        self.optimizer.step()  # 使用梯度更新模型权重
        nodeTraining = {}  # 保存每个节点的权重、梯度和变化量
        for nodeID in self.sortedIDs:
            if nodeID in self.targetNodeIDs: continue  # 目标分支的参数不属于模型训练结果
            parameters = list(self.nodeModules[self.moduleKeys[nodeID]].named_parameters())  # 读取当前节点自身的参数名和值
            if not parameters:
                continue  # 无参数节点不需要显示训练矩阵
            nodeTraining[nodeID] = {"parameters": [{"nodeId": nodeID, "name": name, "weight": serializeValue(parameter.detach(), maxValues), "gradient": serializeValue(parameter.grad.detach() if parameter.grad is not None else torch.zeros_like(parameter), maxValues), "delta": serializeValue((parameter.detach() - beforeWeights[nodeID][name]).detach(), maxValues)} for name, parameter in parameters] }  # 每个参数明确归属节点和真实名称
        return {"loss": float(loss.detach().item()), "prediction": serializeValue(prediction, maxValues), "target": serializeValue(target, maxValues), "nodes": nodeTraining}  # 返回一轮训练的完整可视化快照


# --- 编译持久蓝图模型 ---
def compileBlueprint(blueprint):
    with registry.registryLock:
        return BlueprintModel(blueprint)  # 编译期间阻止插件重载替换注册项


modelCache = {"key": None, "model": None}  # 只缓存最近一张蓝图的编译结果，蓝图不变时复用同一组权重


# --- 计算蓝图业务内容的缓存键 ---
def getModelCacheKey(blueprint):
    businessData = {
        "variables": [(item.get("name"), item.get("value")) for item in blueprint.get("variables", []) if isinstance(item, dict)],  # 变量名称和值决定表达式展开结果
        "nodes": [(node.get("id"), node.get("data", {}).get("opcode"), node.get("data", {}).get("params")) for node in blueprint.get("nodes", [])],  # 节点身份、类型和参数决定模块构建
        "edges": [(edge.get("source"), edge.get("sourceHandle"), edge.get("target"), edge.get("targetHandle")) for edge in blueprint.get("edges", [])],  # 端口连接决定数据流
    }  # 画布坐标和显示名不影响执行，不参与缓存键
    return json.dumps(businessData, sort_keys=True, ensure_ascii=False, default=str)  # 稳定文本键便于直接比较


# --- 蓝图未变化时复用已编译模型 ---
def compileBlueprintCached(blueprint):
    """
    用法：model = compileBlueprintCached(blueprint)  # 连续运行同一蓝图时保持权重不变
    """
    cacheKey = getModelCacheKey(blueprint)  # 只依据业务内容判断蓝图是否变化
    if modelCache["key"] == cacheKey and modelCache["model"] is not None:
        return modelCache["model"]  # 蓝图未变时返回同一模型，随机的只有输入张量
    model = compileBlueprint(blueprint)  # 蓝图变化后重新编译并重新初始化权重
    modelCache["key"] = cacheKey  # 记录当前蓝图键
    modelCache["model"] = model  # 替换唯一缓存槽位
    return model  # 返回新编译模型


# --- 清空模型缓存 ---
def clearModelCache():
    modelCache["key"] = None  # 节点热重载后旧模型持有过期节点类
    modelCache["model"] = None  # 释放模型引用，下一次运行重新编译


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
    shape = module.params.get("out_shape", [2, 4, 8])  # 输入节点缺省沿用全项目统一的小型预览形状
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
def serializeValue(value, maxValues=65536):
    maxValues = max(1, min(int(maxValues), 65536))  # 最多保留256×256个方块，兼顾完整观察和消息体积
    if isinstance(value, torch.Tensor):
        detached = value.detach()  # 反馈数据不应继续持有反向图
        flatValues = detached.cpu().reshape(-1)  # 移到 CPU 后按一维截取固定数量
        shownValues = [serializeNumber(item) for item in flatValues[:maxValues].tolist()]  # 截断后替换JSON不支持的特殊数值
        return {
            "kind": "tensor",  # 前端按稳定类型选择标量、矩阵或高维预览
            "shape": list(detached.shape),  # 原始形状帮助前端理解完整张量
            "dtype": str(detached.dtype).removeprefix("torch."),  # 返回可用于输入协议的 dtype 名称
            "device": str(detached.device),  # 返回执行设备供结果结构保持完整
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
async def run(blueprint, onMessage, onError, inputs=None, maxValues=65536, compiledModel=None):
    startedAt = time.perf_counter()  # 蓝图总耗时从编译前开始计算
    try:
        model = compiledModel or compileBlueprintCached(blueprint)  # 蓝图未变时复用同一模型，让连续观察保持权重稳定
        model.eval()  # 预览默认关闭 dropout 等训练随机行为
        modelInputs = validateModelInputs(model, parseInputs(inputs))  # 输入名称必须对应显式InputNode
        nodeResults = {}  # 当前轮次保存已经传播完成的节点端口值
        nodeDurations = {}  # 当前轮次保存逐节点耗时
        errorCount = 0  # 节点错误只影响自身和依赖它的下游，不终止其他分支
        with torch.no_grad():
            for nodeID in model.sortedIDs:
                incomingEdges = model.edgesByTarget.get(nodeID, [])  # 读取当前节点全部上游连线
                if any(edge.get("source", "") not in nodeResults for edge in incomingEdges):
                    continue  # 任一上游没有结果时跳过当前节点，下游也会自然保持无值
                nodeStartedAt = time.perf_counter()  # 单独记录当前节点计算耗时
                try:
                    inputValues = model._collectInputs(nodeID, nodeResults, modelInputs)  # 只读取已经完成的上游节点结果
                    if model._getOpcode(nodeID) != "input" and any(value is None for value in inputValues.values()):
                        continue  # 上游明确返回空值时不调用当前节点
                    outputValues = model.nodeModules[model.moduleKeys[nodeID]](inputValues)  # 执行当前节点后再进入下游
                    if not isinstance(outputValues, dict):
                        raise BlueprintError("invalidNodeOutput", "节点输出必须是端口对象", {"nodeId": nodeID})  # 图路由依赖命名端口
                except Exception as error:
                    nodeError = error if isinstance(error, BlueprintError) else BlueprintError("nodeExecutionFailed", f"节点执行失败: {error}", {"nodeId": nodeID, "opcode": model._getOpcode(nodeID)})
                    callbackResult = onError(nodeID, nodeError.toData())  # 错误只标记当前节点
                    if inspect.isawaitable(callbackResult):
                        await callbackResult  # 错误图标先更新，再继续计算其他独立分支
                    errorCount += 1
                    continue
                durationMs = (time.perf_counter() - nodeStartedAt) * 1000  # 计算完成后立即得到本节点耗时
                nodeResults[nodeID] = outputValues  # 先写入数据流供下一节点读取
                nodeDurations[nodeID] = durationMs  # 保存最近一次节点耗时
                opcode = model.nodeData[nodeID].get("data", {}).get("opcode", "")  # 节点类型随结果返回
                report = {"opcode": opcode, "inputs": serializeValue(inputValues, maxValues), "outputs": serializeValue(outputValues, maxValues), "durationMs": durationMs}  # 同时反馈真实输入和输出，前端才能展示实际计算
                callbackResult = onMessage(nodeID, report)  # 当前节点计算完成后立即反馈，不等待整图结束
                if inspect.isawaitable(callbackResult):
                    await callbackResult  # WebSocket发送完成后再计算下游节点

        model.lastNodeResults = nodeResults  # 完整传播结束后保存最近结果
        model.lastNodeDurations = nodeDurations  # 完整传播结束后保存最近耗时
        completedOutputIDs = [nodeID for nodeID in model.outputIDs if nodeID in nodeResults]  # 只汇总本轮真正收到上游值的Output节点
        modelOutputs = {nodeID: model._getOutputValue(nodeID, nodeResults[nodeID]) for nodeID in completedOutputIDs}  # 跳过依赖错误节点的Output
        durationMs = (time.perf_counter() - startedAt) * 1000  # 编译和执行都计入用户感知耗时
        status = "completedWithErrors" if errorCount else "succeeded"  # 局部错误仍视为本轮已完成，可继续下一轮
        return {"status": status, "outputs": serializeValue(modelOutputs, maxValues), "outputNodeIds": completedOutputIDs, "errorCount": errorCount, "durationMs": durationMs}  # 返回本轮完整终态
    except Exception as error:
        blueprintError = error if isinstance(error, BlueprintError) else BlueprintError("blueprintFailed", str(error))  # 未分类异常统一收口
        callbackResult = onError(blueprintError.details.get("nodeId", ""), blueprintError.toData())  # 流式反馈结构化错误
        if inspect.isawaitable(callbackResult):
            await callbackResult  # 确保错误先于终态返回
        durationMs = (time.perf_counter() - startedAt) * 1000  # 失败同样报告已消耗时间
        return {"status": "failed", "error": blueprintError.toData(), "outputNodeIds": [], "errorCount": 1, "durationMs": durationMs}  # 由调用方仍发送一次 blueprintComplete
