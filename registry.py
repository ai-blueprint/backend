import torch.nn as nn

nodes = {}
categories = {}

categoriesOrder = ["base", "transform", "activation", "math"]


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
    
    # 第一步：生成优先级分类列表
    priority = [(catId, categories[catId]) for catId in categoriesOrder if catId in categories]  # 按优先级顺序提取存在的分类，生成(分类ID,分类信息)元组列表
    
    # 第二步：生成剩余分类列表
    used = set(catId for catId, _ in priority)  # 提取已使用的分类ID集合，用于后续过滤
    remaining = [(catId, cat) for catId, cat in categories.items() if catId not in used]  # 提取未使用的分类，保持原顺序
    
    # 第三步：拼接列表并转换成字典
    for catId, cat in priority + remaining:  # 遍历合并后的分类列表，先优先级再剩余
        result["categories"][catId] = {k: v for k, v in cat.items() if k != "cls"}  # 添加分类信息到结果，过滤掉cls属性
    
    # 第四步：构建节点数据
    for opcode, node in nodes.items():  # 遍历所有节点
        result["nodes"][opcode] = {k: v for k, v in node.items() if k != "cls"}  # 添加节点信息到结果，过滤掉cls属性
    
    return result  # 返回排序后的结果


def createNode(opcode, nodeId, params):
    """根据opcode创建节点实例"""
    if opcode not in nodes:
        raise ValueError(f"未知节点: {opcode}")
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
