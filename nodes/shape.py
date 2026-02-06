"""
nodes/shape.py - 形状节点组

提供张量形状操作节点：reshape、view、transpose、permute、squeeze、unsqueeze、flatten、unflatten、pad、detach、clone
"""

import torch  # 导入torch用于张量操作
import torch.nn.functional as F  # 导入F用于pad操作
from registry import category, node, BaseNode  # 从registry导入装饰器和基类


# ==================== 分类定义 ====================

category(  # 注册形状分类
    id="shape",  # 分类唯一标识
    label="形状",  # 分类显示名称
    color="#f1af54",  # 分类颜色，紫色
    icon="",  # 分类图标
)


# ==================== 节点定义 ====================


@node(  # 注册reshape节点
    opcode="reshape",  # 节点操作码
    label="reshape",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": "输入"},  # 一个输入端口
        "output": {"out": "输出"},  # 一个输出端口
    },
    params={  # 参数定义
        "shape": {"label": "目标形状", "type": "list", "value": [-1]},  # 目标形状，-1表示自动推断
    },
    description="改变形状，如拆分多头",  # 节点描述
)
class ReshapeNode(BaseNode):  # 继承BaseNode
    """
    reshape形状变换节点
    用法：改变张量形状，元素总数不变 out = x.reshape(shape)
    调用示例：
        输入 x: shape=[batch, seq_len*heads]
        参数 shape=[0, 8, -1] 其中-1自动推断
        输出 out: shape=[batch, 8, seq_len*heads/8]
    """

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        shape = self.params["shape"]["value"]  # 获取目标形状
        out = x.reshape(shape)  # 改变形状
        return {"out": out}  # 返回输出


@node(  # 注册view节点
    opcode="view",  # 节点操作码
    label="view",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": "输入"},  # 一个输入端口
        "output": {"out": "输出"},  # 一个输出端口
    },
    params={  # 参数定义
        "shape": {"label": "目标形状", "type": "list", "value": [-1]},  # 目标形状，-1表示自动推断
    },
    description="改变形状，要求内存连续",  # 节点描述
)
class ViewNode(BaseNode):  # 继承BaseNode
    """
    view形状变换节点
    用法：改变张量形状，要求内存连续 out = x.view(shape)
    调用示例：
        输入 x: shape=[batch, seq_len, heads*dim]
        参数 shape=[0, 0, 8, 64] 其中-1自动推断
        输出 out: shape=[batch, seq_len, 8, 64]
    """

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        shape = self.params["shape"]["value"]  # 获取目标形状
        out = x.view(shape)  # 改变形状（要求连续内存）
        return {"out": out}  # 返回输出


@node(  # 注册transpose节点
    opcode="transpose",  # 节点操作码
    label="transpose",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": "输入"},  # 一个输入端口
        "output": {"out": "输出"},  # 一个输出端口
    },
    params={  # 参数定义
        "dim0": {"label": "维度1", "type": "int", "value": 0, "range": [-10, 10]},  # 要交换的第一个维度
        "dim1": {"label": "维度2", "type": "int", "value": 1, "range": [-10, 10]},  # 要交换的第二个维度
    },
    description="交换两个维度",  # 节点描述
)
class TransposeNode(BaseNode):  # 继承BaseNode
    """
    transpose维度交换节点
    用法：交换两个维度 out = x.transpose(dim0, dim1)
    调用示例：
        输入 x: shape=[batch, seq_len, features]
        参数 dim0=1, dim1=2
        输出 out: shape=[batch, features, seq_len]
    """

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        dim0 = self.params["dim0"]["value"]  # 获取维度1
        dim1 = self.params["dim1"]["value"]  # 获取维度2
        out = x.transpose(dim0, dim1)  # 交换两个维度
        return {"out": out}  # 返回输出


@node(  # 注册permute节点
    opcode="permute",  # 节点操作码
    label="permute",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": "输入"},  # 一个输入端口
        "output": {"out": "输出"},  # 一个输出端口
    },
    params={  # 参数定义
        "dims": {"label": "维度顺序", "type": "list", "value": [0, 2, 1]},  # 新的维度排列顺序
    },
    description="任意重排所有维度顺序",  # 节点描述
)
class PermuteNode(BaseNode):  # 继承BaseNode
    """
    permute维度重排节点
    用法：任意重排所有维度 out = x.permute(dims)
    调用示例：
        输入 x: shape=[batch, seq_len, heads, dim]
        参数 dims=[0, 2, 1, 3]
        输出 out: shape=[batch, heads, seq_len, dim]
    """

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        dims = self.params["dims"]["value"]  # 获取维度顺序
        out = x.permute(dims)  # 重排维度
        return {"out": out}  # 返回输出


@node(  # 注册squeeze节点
    opcode="squeeze",  # 节点操作码
    label="squeeze",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": "输入"},  # 一个输入端口
        "output": {"out": "输出"},  # 一个输出端口
    },
    params={  # 参数定义
        "dim": {"label": "维度", "type": "int", "value": -1, "range": [-10, 10]},  # 要去掉的维度
    },
    description="去掉大小为1的维度",  # 节点描述
)
class SqueezeNode(BaseNode):  # 继承BaseNode
    """
    squeeze压缩维度节点
    用法：去掉指定位置大小为1的维度 out = x.squeeze(dim)
    调用示例：
        输入 x: shape=[batch, 1, features]
        参数 dim=1
        输出 out: shape=[batch, features]
    """

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        dim = self.params["dim"]["value"]  # 获取维度
        out = x.squeeze(dim)  # 去掉大小为1的维度
        return {"out": out}  # 返回输出


@node(  # 注册unsqueeze节点
    opcode="unsqueeze",  # 节点操作码
    label="unsqueeze",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": "输入"},  # 一个输入端口
        "output": {"out": "输出"},  # 一个输出端口
    },
    params={  # 参数定义
        "dim": {"label": "维度", "type": "int", "value": 0, "range": [-10, 10]},  # 要插入的维度位置
    },
    description="增加一个大小为1的维度",  # 节点描述
)
class UnsqueezeNode(BaseNode):  # 继承BaseNode
    """
    unsqueeze扩展维度节点
    用法：在指定位置插入一个大小为1的维度 out = x.unsqueeze(dim)
    调用示例：
        输入 x: shape=[batch, features]
        参数 dim=1
        输出 out: shape=[batch, 1, features]
    """

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        dim = self.params["dim"]["value"]  # 获取维度
        out = x.unsqueeze(dim)  # 插入大小为1的维度
        return {"out": out}  # 返回输出


@node(  # 注册flatten节点
    opcode="flatten",  # 节点操作码
    label="flatten",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": "输入"},  # 一个输入端口
        "output": {"out": "输出"},  # 一个输出端口
    },
    params={  # 参数定义
        "start_dim": {"label": "起始维度", "type": "int", "value": 1, "range": [-10, 10]},  # 从哪个维度开始压平
        "end_dim": {"label": "结束维度", "type": "int", "value": -1, "range": [-10, 10]},  # 到哪个维度结束压平
    },
    description="多个维度压成一个",  # 节点描述
)
class FlattenNode(BaseNode):  # 继承BaseNode
    """
    flatten压平节点
    用法：将start_dim到end_dim之间的维度压成一个 out = x.flatten(start_dim, end_dim)
    调用示例：
        输入 x: shape=[batch, channels, height, width]
        参数 start_dim=1, end_dim=-1
        输出 out: shape=[batch, channels*height*width]
    """

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        start_dim = self.params["start_dim"]["value"]  # 获取起始维度
        end_dim = self.params["end_dim"]["value"]  # 获取结束维度
        out = x.flatten(start_dim, end_dim)  # 压平维度
        return {"out": out}  # 返回输出


@node(  # 注册unflatten节点
    opcode="unflatten",  # 节点操作码
    label="unflatten",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": "输入"},  # 一个输入端口
        "output": {"out": "输出"},  # 一个输出端口
    },
    params={  # 参数定义
        "dim": {"label": "维度", "type": "int", "value": 1, "range": [-10, 10]},  # 要展开的维度
        "sizes": {"label": "展开形状", "type": "list", "value": [8, 64]},  # 展开后的形状
    },
    description="一个维度展开成多个",  # 节点描述
)
class UnflattenNode(BaseNode):  # 继承BaseNode
    """
    unflatten展开节点
    用法：将一个维度展开成多个 out = x.unflatten(dim, sizes)
    调用示例：
        输入 x: shape=[batch, 512]
        参数 dim=1, sizes=[8, 64]
        输出 out: shape=[batch, 8, 64]
    """

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        dim = self.params["dim"]["value"]  # 获取维度
        sizes = self.params["sizes"]["value"]  # 获取展开形状
        out = x.unflatten(dim, sizes)  # 展开维度
        return {"out": out}  # 返回输出


@node(  # 注册pad节点
    opcode="pad",  # 节点操作码
    label="pad",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": "输入"},  # 一个输入端口
        "output": {"out": "输出"},  # 一个输出端口
    },
    params={  # 参数定义
        "padding": {"label": "填充量", "type": "list", "value": [0, 0, 0, 0]},  # 各维度的填充量
        "mode": {"label": "填充模式", "type": "enum", "value": "constant", "options": {"constant": "常数填充", "reflect": "反射填充", "replicate": "复制填充", "circular": "循环填充"}},  # 填充模式
        "value": {"label": "填充值", "type": "float", "value": 0.0, "range": [-1e6, 1e6]},  # 常数填充时的值
    },
    description="边缘填充值，序列对齐用",  # 节点描述
)
class PadNode(BaseNode):  # 继承BaseNode
    """
    pad填充节点
    用法：在张量边缘填充值 out = F.pad(x, padding, mode, value)
    调用示例：
        输入 x: shape=[batch, seq_len, features]
        参数 padding=[0, 0, 1, 1] 表示最后两个维度各填充1
        输出 out: shape=[batch, seq_len+2, features]
    """

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        padding = self.params["padding"]["value"]  # 获取填充量
        mode = self.params["mode"]["value"]  # 获取填充模式
        value = self.params["value"]["value"]  # 获取填充值
        out = F.pad(x, padding, mode=mode, value=value)  # 边缘填充
        return {"out": out}  # 返回输出


@node(  # 注册detach节点
    opcode="detach",  # 节点操作码
    label="detach",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": "输入"},  # 一个输入端口
        "output": {"out": "输出"},  # 一个输出端口
    },
    params={},  # 无参数
    description="切断梯度传播",  # 节点描述
)
class DetachNode(BaseNode):  # 继承BaseNode
    """
    detach梯度切断节点
    用法：切断梯度传播，返回不参与反向传播的张量 out = x.detach()
    调用示例：
        输入 x: shape=[任意形状]
        输出 out: shape=[与输入相同]，但不再追踪梯度
    """

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        out = x.detach()  # 切断梯度
        return {"out": out}  # 返回输出


@node(  # 注册clone节点
    opcode="clone",  # 节点操作码
    label="clone",  # 节点显示名称
    ports={  # 端口定义
        "input": {"x": "输入"},  # 一个输入端口
        "output": {"out": "输出"},  # 一个输出端口
    },
    params={},  # 无参数
    description="深拷贝，独立副本",  # 节点描述
)
class CloneNode(BaseNode):  # 继承BaseNode
    """
    clone深拷贝节点
    用法：创建张量的独立副本 out = x.clone()
    调用示例：
        输入 x: shape=[任意形状]
        输出 out: shape=[与输入相同]，完全独立的副本
    """

    def compute(self, input):  # 计算方法
        x = input.get("x")  # 获取输入张量
        out = x.clone()  # 深拷贝
        return {"out": out}  # 返回输出
