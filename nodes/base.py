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
    params={
        "out_shape": {"label": "输出形状", "type": "list", "value": [2, 4, 8]}
    },  # 参数定义
    description="生成随机张量作为输入",  # 节点描述
)
class InputNode(BaseNode):  # 继承BaseNode
    def compute(self, input):  # 计算方法
        if "value" in input:
            return {"out": input["value"]}  # 显式模型输入优先，训练和导出不会被随机值覆盖
        shape = self.params.get("out_shape", [2, 4, 8])  # 读取扁平参数里的输出形状
        return {"out": torch.rand(shape) * 2 - 1}  # 返回[-1, 1)均匀分布随机张量


@node(  # 注册输出节点
    opcode="output",  # 节点操作码
    label="输出",  # 节点显示名称
    ports={"input": {"in": "预测值", "target": "目标值"}, "output": {}},  # 输出节点同时接收预测值和可选目标值
    description="接收预测结果，可连接目标值进入训练模式",  # 节点描述
)
class OutputNode(BaseNode):  # 继承BaseNode
    def compute(self, input):  # 计算方法
        prediction = input.get("prediction", input.get("in", None))  # 保留旧in端口，同时允许未来协议使用prediction名称
        target = input.get("target", None)  # 目标端口没有连接时保持普通前向传播
        return {"out": prediction, "target": target}  # 同时保留预测和目标供运行与训练协议读取


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
