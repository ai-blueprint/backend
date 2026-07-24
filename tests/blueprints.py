"""测试蓝图构造器，集中表达测试所需的最小图数据。"""


def linearBlueprint(inputShape=None):
    inputShape = inputShape or [2, 3]  # 默认形状兼顾快速执行和批量语义
    return {
        "nodes": [
            {"id": "input.with.dot", "data": {"opcode": "input", "params": {"out_shape": inputShape}}},
            {"id": "linear-1", "data": {"opcode": "linear", "params": {"in_features": inputShape[-1], "out_features": 2, "bias": True}}},
            {"id": "output-1", "data": {"opcode": "output", "params": {}}},
        ],
        "edges": [
            {"source": "input.with.dot", "sourceHandle": "out", "target": "linear-1", "targetHandle": "x"},
            {"source": "linear-1", "sourceHandle": "out", "target": "output-1", "targetHandle": "in"},
        ],
    }  # 输入、可训练层和显式输出组成端到端最小模型
