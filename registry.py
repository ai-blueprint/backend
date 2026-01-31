import torch.nn as nn

nodes = {}
categories = {}


def registerCategory(id, label, color, icon):
    category = {}
    category["label"] = label
    category["color"] = color
    category["icon"] = icon
    category["nodes"] = []
    categories[id] = category


def registerNode(opcode, label, ports, params, cls):
    node = {}
    node["opcode"] = opcode
    node["label"] = label
    node["ports"] = ports
    # 把params里面的所有param的default键换成value键
    for param in params.values():
        if "default" in param:
            param["value"] = param["default"]
            del param["default"]

    node["params"] = params
    node["cls"] = cls
    nodes[opcode] = node
    categories[list(categories.keys())[-1]]["nodes"].append(opcode)


def getAllForFrontend():
    result = {"categories": categories, "nodes": {}}
    for opcode, node in nodes.items():
        result["nodes"][opcode] = {k: v for k, v in node.items() if k != "cls"}
    return result


def createNode(opcode, nodeId, params):
    """根据opcode创建节点实例"""
    if opcode not in nodes:
        raise ValueError(f"未知节点: {opcode}")
    cls = nodes[opcode]["cls"]
    return cls(nodeId, params)


def category(id="", label="", color="#8992eb", icon=""):
    registerCategory(id, label, color, icon)


def node(opcode="", label="", ports={}, params={}):
    def decorator(cls):
        registerNode(opcode, label, ports, params, cls)
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
