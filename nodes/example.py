"""
示例节点定义

演示如何使用装饰器定义节点和分类。
"""

from decorators import category, node  # 从装饰器导入category, node
import torch


@category(  # @category
    id="example_category",  # id="example_category"
    name="示例分类",  # name="示例分类"
    color="#FFB6C1"  # color="#FFB6C1"
)
def example_category():  # 定义示例分类
    """示例分类定义"""
    pass


@node(  # @node
    id="example_node",  # id="example_node"
    name="示例节点",  # name="示例节点"
    inputs=["x", "y"],  # inputs=["x", "y"]
    outputs=["result"],  # outputs=["result"]
    params={"数字参数": 1, "布尔参数": False}  # params={"数字参数": 1, "布尔参数": False}
)
def example_node():  # 定义示例节点
    """
    示例节点定义

    演示节点的完整结构：infer_shape、build、compute三个函数
    """

    def infer_shape(input_shapes, params):  # infer_shape方法
        """
        形状推断函数

        参数:
            input_shapes: 输入形状字典 {"x": [batch, dim], "y": [batch, dim]}
            params: 节点参数 {"数字参数": 1, "布尔参数": False}

        返回:
            输出形状字典 {"result": [batch, dim]}
        """
        # 简单示例：输出形状与第一个输入相同
        x_shape = input_shapes.get('x', [1, 10])
        return {"result": x_shape}

    def build(input_shapes, params):  # build方法
        """
        构建层实例

        参数:
            input_shapes: 输入形状字典
            params: 节点参数

        返回:
            层实例（可以是None、nn.Module或任何自定义对象）
        """
        # 示例：不需要构建特殊层，返回None
        # 如果需要构建nn.Module，可以返回：
        # return torch.nn.Linear(input_shapes['x'][-1], output_dim)
        return None

    def compute(inputs, layer):  # compute方法
        """
        执行计算

        参数:
            inputs: 输入数据字典 {"x": tensor, "y": tensor}
            layer: 构建的层实例（来自build函数）

        返回:
            输出数据，可以是：
            - 字典：{"result": tensor}
            - 单个值：会自动包装为 {"result": value}
            - 元组：按端口顺序映射
        """
        # 示例：将两个输入相加
        x = inputs.get('x')
        y = inputs.get('y')

        if x is None or y is None:
            # 如果输入不完整，返回默认值
            return {"result": torch.tensor([0.0])}

        # 确保是张量
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        # 执行计算
        result = x + y

        return {"result": result}

    return infer_shape, build, compute  # 返回infer_shape、build、compute这三个func
