"""
nodes/base.py - 基础节点组

提供基础的输入/输出/调试/常量节点
"""

import torch  # 导入torch用于张量操作
from registry import category, node, BaseNode  # 从registry导入装饰器和基类


# ==================== 分类定义 ====================

category(  # 注册基础分类
    id="base",  # 分类唯一标识
    label="基础",  # 分类显示名称
    color="#8B92E5",  # 分类颜色
    icon="",  # 分类图标
)


# ==================== 节点定义 ====================


@node(  # 注册输入节点
    opcode="input",  # 节点操作码
    label="输入",  # 节点显示名称
    ports={"input": {}, "output": {"out": ""}},  # 端口定义，输入节点没有输入端口
    params={"out_shape": {"label": "输出形状", "type": "list", "value": [2, 4, 10]}},  # 参数定义
    description="生成随机张量作为输入",  # 节点描述
)
class InputNode(BaseNode):  # 继承BaseNode
    def compute(self, input):  # 计算方法
        shape = self.params.get("out_shape").get("value")  # 使用字典键访问
        return {"out": torch.rand(shape)}  # 返回随机张量


@node(  # 注册输出节点
    opcode="output",  # 节点操作码
    label="输出",  # 节点显示名称
    ports={"input": {"in": ""}, "output": {}},  # 端口定义，输出节点没有输出端口
    description="接收并打印最终结果",  # 节点描述
)
class OutputNode(BaseNode):  # 继承BaseNode
    def compute(self, input):  # 计算方法
        value = input.get("in", None)  # 获取输入值
        print(f"[Output] 最终输出: {value}")  # 打印最终结果
        return {}  # 返回空字典，没有输出端口


@node(  # 注册调试节点
    opcode="debug",  # 节点操作码
    label="调试输出",  # 节点显示名称
    ports={"input": {"x": ""}, "output": {"out": ""}},  # 端口定义，不需要显示文字
    description="打印张量的形状和类型",  # 节点描述
)
class DebugNode(BaseNode):  # 继承BaseNode
    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入值
        print(f"调试输出：shape={x.shape}, dtype={x.dtype}")  # 打印形状和类型
        return {"out": x}  # 透传输入
