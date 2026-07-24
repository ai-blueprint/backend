"""
nodes/example.py - 示例节点定义
"""

import torch.nn as nn  # 导入torch.nn模块
from registry import category, node, BaseNode  # 从装饰器模块导入category和node装饰器和BaseNode基类


# 定义一个示例分类
category(  # 调用category装饰器注册分类
    id="example_category",  # 分类唯一标识
    label="示例",  # 分类显示名称
    color="#8992eb",  # 分类颜色
    icon="",  # 分类图标，要求是base64格式字符串
)


@node(
    opcode="example_node",  # 节点操作码，唯一标识
    label="示例节点",  # 节点显示名称
    ports={"input": {"x": "输入1", "y": "输入2"}, "output": {"result": "输出"}},  # 输入输出端口定义
    params={
        "int_param": {"label": "整数参数示例", "type": "int", "value": 8, "range": [1, 1024]},  # 默认宽度匹配Input最后一维
        "float_param": {"label": "浮点数参数示例", "type": "float", "value": 0.5, "range": [0, 1]},  # 浮点数参数示例，range可选
        "bool_param": {"label": "布尔参数示例", "type": "bool", "value": True},  # 布尔参数示例
        "str_param": {"label": "字符串参数示例", "type": "str", "value": "默认字符串"},  # 字符串参数示例
        "list_param": {"label": "列表参数示例", "type": "list", "value": [2, 4, 8]},  # 列表参数示例，可以用于形状定义
        "enum_param": {"label": "选项参数示例", "type": "enum", "value": "option1", "options": {"option1": "选项1", "option2": "选项2", "option3": "选项3"}},  # 选项参数示例
    },  # 节点参数定义
    description="演示各种参数类型的用法",  # 节点描述
)
class ExampleNode(BaseNode):
    def build(self):  # 构建示例层
        width = self.params.get("int_param", 8)  # 默认宽度与Input特征维一致
        use_bias = self.params.get("bool_param", True)  # 布尔参数控制线性层偏置
        self.example_act = nn.ReLU()  # 激活层保持示例输出非负
        self.linear = nn.Linear(width, width, bias=use_bias)  # 线性层只变换最后一维

    def compute(self, input):  # 计算示例结果
        combined = input["x"] + input["y"]  # 合并两个同形输入
        out = self.linear(combined)  # 对最后一维执行宽度为8的线性变换
        return {"result": self.example_act(out)}  # 激活后返回结果
