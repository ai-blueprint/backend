import torch.nn as nn

nodes = {}
categories = {}

categoriesOrder = ["base", "transform", "activation", "math"]


def clearAll():  # 清空注册表，热重载时调用
    nodes.clear()  # 清空节点字典
    categories.clear()  # 清空分类字典


def registerCategory(id, label, color, icon):
    category = {}
    category["label"] = label
    category["color"] = color
    category["icon"] = icon
    category["nodes"] = []
    categories[id] = category


def registerNode(opcode, label, ports, params, description, cls):
    node = {}
    node["opcode"] = opcode
    node["label"] = label
    node["ports"] = ports
    node["params"] = params
    node["description"] = description
    node["cls"] = cls
    nodes[opcode] = node
    categories[list(categories.keys())[-1]]["nodes"].append(opcode)


def getAllForFrontend():
    # 根据categoriesOrder排序，categoriesOrder是优先级顺序，不是所有分类都要包含
    result = {"categories": {}, "nodes": {}}  # 初始化返回结果字典，包含分类和节点两个部分

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
        if opcode not in visibleOpcodes: continue  # 跳过隐藏分类下的节点
        result["nodes"][opcode] = {k: v for k, v in node.items() if k != "cls"}  # 添加节点信息到结果，过滤掉cls属性

    return result  # 返回排序后的结果


def validateParams(opcode, params):
    """根据注册定义校验并修正参数值，返回修正后的params"""
    definition = nodes[opcode].get("params", {})  # 获取注册时的参数定义
    validated = {}  # 校验后的参数字典
    for key, value in params.items():  # 遍历前端传来的每个参数
        if isinstance(value, dict) and "value" in value:  # 前端传来的是完整定义字典，提取实际值
            value = value["value"]  # 取出value字段
        spec = definition.get(key)  # 获取该参数的注册定义
        if spec is None:  # 注册定义中不存在此参数，跳过
            validated[key] = value  # 原样保留
            continue
        paramType = spec.get("type", "")  # 获取参数类型
        paramRange = spec.get("range")  # 获取range限制
        paramOptions = spec.get("options")  # 获取enum选项
        defaultValue = spec.get("value")  # 获取默认值
        if paramType == "int":  # 整数类型校验
            if not isinstance(value, (int, float)) or isinstance(value, bool):  # 类型不兼容（bool是int子类需排除）
                print(f"参数类型错误: {opcode}.{key} 期望int，实际{type(value).__name__}={value}，回退默认值{defaultValue}")  # 类型不符报错
                value = defaultValue  # 回退默认值
            else:
                value = int(value)  # 强转int
            if paramRange and len(paramRange) == 2:  # 有range则clamp
                clamped = max(paramRange[0], min(paramRange[1], value))  # 钳位到[min, max]
                if clamped != value:  # 值被修正了
                    print(f"参数越界修正: {opcode}.{key} 值{value}超出范围{paramRange}，修正为{clamped}")  # 越界报错
                value = clamped  # 使用修正后的值
        elif paramType == "float":  # 浮点类型校验
            if not isinstance(value, (int, float)) or isinstance(value, bool):  # 类型不兼容
                print(f"参数类型错误: {opcode}.{key} 期望float，实际{type(value).__name__}={value}，回退默认值{defaultValue}")  # 类型不符报错
                value = defaultValue  # 回退默认值
            else:
                value = float(value)  # 强转float
            if paramRange and len(paramRange) == 2:  # 有range则clamp
                clamped = max(paramRange[0], min(paramRange[1], value))  # 钳位到[min, max]
                if clamped != value:  # 值被修正了
                    print(f"参数越界修正: {opcode}.{key} 值{value}超出范围{paramRange}，修正为{clamped}")  # 越界报错
                value = clamped  # 使用修正后的值
        elif paramType == "bool":  # 布尔类型校验
            if not isinstance(value, bool):  # 类型不兼容
                print(f"参数类型错误: {opcode}.{key} 期望bool，实际{type(value).__name__}={value}，回退默认值{defaultValue}")  # 类型不符报错
                value = defaultValue  # 回退默认值
        elif paramType == "str":  # 字符串类型校验
            if not isinstance(value, str):  # 类型不兼容
                print(f"参数类型错误: {opcode}.{key} 期望str，实际{type(value).__name__}={value}，回退默认值{defaultValue}")  # 类型不符报错
                value = defaultValue  # 回退默认值
        elif paramType == "enum" and paramOptions:  # 枚举类型校验
            if value not in paramOptions:  # 值不在合法选项中
                print(f"参数选项无效: {opcode}.{key} 值{value}不在选项{list(paramOptions.keys())}中，回退默认值{defaultValue}")  # 选项无效报错
                value = defaultValue  # 回退到默认值
        elif paramType == "list":  # 列表类型校验
            if not isinstance(value, list):  # 类型不兼容
                print(f"参数类型错误: {opcode}.{key} 期望list，实际{type(value).__name__}={value}，回退默认值{defaultValue}")  # 类型不符报错
                value = defaultValue  # 回退默认值
            else:  # 是list，校验元素
                cleaned = []  # 清洗后的列表
                for i, item in enumerate(value):  # 遍历每个元素
                    if isinstance(item, (int, float)) and not isinstance(item, bool):  # 元素是数字
                        cleaned.append(item)  # 保留
                    else:  # 元素不是数字
                        print(f"参数元素无效: {opcode}.{key}[{i}] 期望数字，实际{type(item).__name__}={item}，回退默认值{defaultValue}")  # 报错
                        value = defaultValue  # 整个list回退默认值
                        break  # 跳出循环
                else:  # 全部通过
                    value = cleaned  # 使用清洗后的列表
        validated[key] = value  # 存入校验后的字典
    return validated  # 返回校验后的参数


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
