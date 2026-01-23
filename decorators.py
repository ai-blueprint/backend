"""
节点注册装饰器模块

提供 @category 和 @node 装饰器，用于声明式地注册节点。
对应开发目标.txt L134-147
"""

from typing import Any, Callable, Dict, List, Optional

import registry


# ==================== 全局状态 ====================

_current_category: Optional[str] = None  # 当前激活的分类


# ==================== 装饰器 ====================

def category(  # @category装饰器
    id: str,  # 参数：id
    name: str,  # 参数：name
    color: str = "#888888"  # 参数：color
) -> Callable:
    """
    分类装饰器

    用于定义一个节点分类，后续的 @node 装饰器会自动归入此分类。

    参数:
        id: 分类唯一标识符
        name: 分类显示名称
        color: 分类主题颜色（十六进制）

    返回:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:  # 返回装饰器函数
        global _current_category

        # 验证参数
        if not id or not isinstance(id, str):
            raise ValueError("分类 id 必须是非空字符串")
        if not name or not isinstance(name, str):
            raise ValueError("分类 name 必须是非空字符串")

        registry.register_category(id, name, color)  # 调用 registry.register_category

        # 设置当前分类
        _current_category = id

        return func

    return decorator


def node(  # @node装饰器
    id: str,  # 参数：id
    name: str,  # 参数：name
    category: Optional[str] = None,  # 参数：category
    inputs: Optional[List[str]] = None,  # 参数：inputs
    outputs: Optional[List[str]] = None,  # 参数：outputs
    params: Optional[Dict[str, Any]] = None  # 参数：params
) -> Callable:
    """
    节点装饰器

    用于注册一个节点定义。节点函数必须返回 (infer, build, compute) 三元组。

    参数:
        id: 节点唯一标识符
        name: 节点显示名称
        category: 所属分类ID（如果为None，使用当前分类）
        inputs: 输入端口名称列表
        outputs: 输出端口名称列表
        params: 参数定义 {"param_name": default_value}

    返回:
        装饰器函数

    示例:
        @node(
            id="linear",
            name="全连接层",
            inputs=["x"],
            outputs=["out"],
            params={"in_features": 128, "out_features": 64}
        )
        def linear_layer():
            def infer(input_shapes, params): ...
            def build(input_shapes, params): ...
            def compute(x, layer): ...
            return infer, build, compute
    """
    def decorator(func: Callable) -> Callable:  # 返回装饰器函数
        global _current_category

        # 验证参数
        if not id or not isinstance(id, str):
            raise ValueError("节点 id 必须是非空字符串")
        if not name or not isinstance(name, str):
            raise ValueError("节点 name 必须是非空字符串")

        # 确定分类
        node_category = category if category is not None else _current_category
        if node_category is None:
            raise ValueError(
                f"节点 {id} 没有定义在任何分类下！"
                f"请先用 @category 定义分类，或在 @node 中指定 category 参数。"
            )

        # 规范化端口和参数
        node_inputs = inputs if inputs is not None else []
        node_outputs = outputs if outputs is not None else []
        node_params = params if params is not None else {}

        registry.register_node(  # 调用 registry.register_node
            id=id,
            name=name,
            category=node_category,
            inputs=node_inputs,
            outputs=node_outputs,
            params=node_params,
            func=func  # 接收 factory 函数
        )

        return func  # 返回 factory

    return decorator


# ==================== 辅助函数 ====================

def set_current_category(cat_id: str):
    """
    手动设置当前分类（用于测试或特殊场景）

    参数:
        cat_id: 分类ID
    """
    global _current_category
    if cat_id not in registry._categories:
        raise ValueError(f"分类 {cat_id} 不存在")
    _current_category = cat_id


def get_current_category() -> Optional[str]:
    """获取当前激活的分类ID"""
    return _current_category


def reset_current_category():
    """重置当前分类（用于测试）"""
    global _current_category
    _current_category = None
