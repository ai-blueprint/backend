import torch.nn as nn

nodes = {}
categories = {}

categoriesOrder = ["base", "transform", "activation", "math"]


def clearAll():  # 清空注册表，热重载时调用
    nodes.clear()  # 清空节点字典
    categories.clear()  # 清空分类字典


def registerCategory(id, label, color, icon):
    categories[id] = {"label": label, "color": color, "icon": icon, "nodes": []}


def registerNode(opcode, label, ports, params, description, cls):
    nodes[opcode] = {"opcode": opcode, "label": label, "ports": ports, "params": params, "description": description, "cls": cls}
    categories[list(categories.keys())[-1]]["nodes"].append(opcode)


def getAllForFrontend():
    # 根据categoriesOrder排序，categoriesOrder是优先级顺序，不是所有分类都要包含
    result = {
        "categories": {},
        "nodes": {},
    }  # 初始化返回结果字典，包含分类和节点两个部分

    hiddenCategories = {"example_category"}  # 前端默认隐藏的分类集合

    # 第一步：生成优先级分类列表
    priority = [(catId, categories[catId]) for catId in categoriesOrder if catId in categories and catId not in hiddenCategories]  # 按优先级顺序提取存在且非隐藏的分类

    # 第二步：生成剩余分类列表
    used = set(catId for catId, _ in priority)  # 提取已使用的分类ID集合，用于后续过滤
    remaining = [(catId, cat) for catId, cat in categories.items() if catId not in used and catId not in hiddenCategories]  # 提取未使用且非隐藏的分类

    # 第三步：拼接列表并转换成字典
    for catId, cat in priority + remaining:  # 遍历合并后的分类列表，先优先级再剩余
        result["categories"][catId] = {k: v for k, v in cat.items() if k != "cls"}  # 添加分类信息到结果，过滤掉cls属性

    # 第四步：收集所有可见节点的opcode
    visibleOpcodes = set()  # 可见节点opcode集合
    for catId, cat in result["categories"].items():  # 遍历可见分类
        visibleOpcodes.update(cat.get("nodes", []))  # 收集该分类下的所有节点opcode

    # 第五步：构建节点数据（仅包含可见分类下的节点）
    for opcode, node in nodes.items():  # 遍历所有节点
        if opcode not in visibleOpcodes:
            continue  # 跳过隐藏分类下的节点
        result["nodes"][opcode] = {k: v for k, v in node.items() if k != "cls"}  # 添加节点信息到结果，过滤掉cls属性

    return result  # 返回排序后的结果


def validateParams(opcode, params):
    """根据注册定义做通用参数校验，统一返回扁平值字典。"""
    definition = nodes[opcode].get("params", {})  # 获取节点参数定义
    if not isinstance(params, dict):
        params = {}  # 入参不是字典时使用空字典兜底

    validated = {}  # 存储校验后的扁平参数

    for key, spec in definition.items():  # 先遍历定义，保证缺失参数也有默认值
        defaultValue = spec.get("value")  # 默认值
        value = params.get(key, defaultValue)  # 前端传值优先，没有就用默认值

        if isinstance(value, dict) and "value" in value:
            value = value.get("value")  # 兼容旧格式参数对象

        if value is None:
            value = defaultValue  # 空值回退默认值

        options = spec.get("options")  # 选项限制
        if options and value not in options:
            print(f"参数选项无效：{opcode}.{key}={value}，回退默认值{defaultValue}")  # 打印修正日志
            value = defaultValue  # 非法选项回退默认值

        paramRange = spec.get("range")  # 范围限制
        canClamp = isinstance(value, (int, float)) and not isinstance(value, bool)  # 只对数值做范围修正
        hasRange = isinstance(paramRange, (list, tuple)) and len(paramRange) == 2  # 范围配置必须是两个边界值
        if canClamp and hasRange:
            minValue = paramRange[0]  # 最小边界
            maxValue = paramRange[1]  # 最大边界
            corrected = max(minValue, min(maxValue, value))  # 执行夹逼修正
            if corrected != value:
                print(f"参数越界修正：{opcode}.{key}={value}，修正为{corrected}")  # 打印修正日志
            value = corrected  # 写入修正结果

        validated[key] = value  # 保存当前参数结果

    for key, value in params.items():  # 保留定义外的扩展参数，保证参数适配宽容
        if key in validated:
            continue  # 已处理过的键直接跳过
        if isinstance(value, dict) and "value" in value:
            value = value.get("value")  # 兼容旧格式扩展参数
        validated[key] = value  # 原样保留扩展参数

    return validated  # 返回扁平参数字典


def createNode(opcode, nodeId, params):
    """根据opcode创建节点实例，创建前校验参数"""
    if opcode not in nodes:
        raise ValueError(f"未知节点: {opcode}")
    params = validateParams(opcode, params)  # 校验并修正参数值
    cls = nodes[opcode]["cls"]
    return cls(nodeId, params)


def category(id="", label="", color="#8992eb", icon=""):
    registerCategory(id, label, color, icon)


def node(opcode="", label="", ports={}, params={}, description=""):
    def decorator(cls):
        registerNode(opcode, label, ports, params, description, cls)
        return cls

    return decorator


class BaseNode(nn.Module):
    def __init__(self, nodeId, params):
        super().__init__()
        self.nodeId = nodeId
        self.params = params
        self.build()

    def build(self):
        pass

    def compute(self, input):
        raise NotImplementedError("必须实现compute")

    def forward(self, input):
        out = self.compute(input)
        # 占位，到时候做值存储和转发操作
        return out
