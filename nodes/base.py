"""
nodes/base.py - 基础节点组

提供基础的输入/输出/调试/常量节点
"""

import torch  # 导入torch用于张量操作
from registry import category, node, BaseNode  # 从registry导入装饰器和基类


# ==================== 分类定义 ====================

category(  # 注册基础分类
    id="basic",  # 分类唯一标识
    label="基础",  # 分类显示名称
    color="#8B92E5",  # 分类颜色
    icon="",  # 分类图标
)


# ==================== 节点定义 ====================


@node(  # 注册输入节点
    opcode="input",  # 节点操作码
    label="输入",  # 节点显示名称
    ports={"input": {}, "output": {"out": "输出"}},  # 端口定义，输入节点没有输入端口
    params={"输出维度": {"label": "输出维度", "type": "list", "default": [1, 10]}},  # 参数定义
)
class InputNode(BaseNode):  # 继承BaseNode
    def build(self):  # 构建方法，在__init__时自动调用
        shape = self.params.get("输出维度", [1, 10])  # 获取输出维度参数
        self.data = torch.rand(*shape)  # 预生成随机张量

    def compute(self, input):  # 计算方法
        return {"out": self.data}  # 返回预生成的张量


@node(  # 注册输出节点
    opcode="output",  # 节点操作码
    label="输出",  # 节点显示名称
    ports={"input": {"in": "输入"}, "output": {}},  # 端口定义，输出节点没有输出端口
    params={},  # 无参数
)
class OutputNode(BaseNode):  # 继承BaseNode
    def build(self):  # 构建方法
        pass  # 输出节点不需要构建任何东西

    def compute(self, input):  # 计算方法
        value = input.get("in", None)  # 获取输入值
        print(f"[Output] 最终输出: {value}")  # 打印最终结果
        return {}  # 返回空字典，没有输出端口


@node(  # 注册常量节点
    opcode="constant",  # 节点操作码
    label="常量",  # 节点显示名称
    ports={"input": {}, "output": {"out": "输出"}},  # 端口定义
    params={"value": {"label": "常量值", "type": "float", "default": 0.0}},  # 参数定义
)
class ConstantNode(BaseNode):  # 继承BaseNode
    def build(self):  # 构建方法
        value = self.params.get("value", 0.0)  # 获取常量值
        self.tensor = torch.tensor([value], dtype=torch.float32)  # 创建张量

    def compute(self, input):  # 计算方法
        return {"out": self.tensor}  # 返回常量张量


@node(  # 注册调试节点
    opcode="debug",  # 节点操作码
    label="调试输出",  # 节点显示名称
    ports={"input": {"x": "输入"}, "output": {"out": "输出"}},  # 端口定义
    params={"label": {"label": "标签", "type": "str", "default": "debug"}},  # 参数定义
)
class DebugNode(BaseNode):  # 继承BaseNode
    def build(self):  # 构建方法
        self.label = self.params.get("label", "debug")  # 获取标签参数

    def compute(self, input):  # 计算方法
        x = input.get("x", None)  # 获取输入x
        if x is not None and hasattr(x, "shape"):  # 如果x是张量
            print(f"[{self.label}] shape={x.shape}, dtype={x.dtype}")  # 打印形状和类型
        else:  # 如果x不是张量
            print(f"[{self.label}] value={x}, type={type(x)}")  # 打印值和类型
        return {"out": x}  # 透传输入
