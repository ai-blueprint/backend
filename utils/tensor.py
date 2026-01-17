"""
张量操作工具模块

提供张量的提取、转换、形状处理等功能。
所有函数都设计为容错的，能够处理各种边界情况。
"""

import torch
from typing import Any, Dict, List, Optional, Union

# ==================== 类型别名 ====================
TensorLike = Union[torch.Tensor, List, None]
ShapeLike = Union[List[int], tuple, None]


def extract_single_input(
    inputs: Any, 
    port_name: str = "x",
    *,
    fallback_first: bool = True
) -> Optional[torch.Tensor]:
    """
    从输入中智能提取单个张量
    
    设计原则：
    1. 无论输入什么格式，都尽最大努力返回有效张量
    2. 优先按端口名匹配，其次取第一个值
    3. 绝不抛出异常，最坏情况返回 None
    
    参数:
        inputs: 可以是 dict、Tensor、list 或 None
        port_name: 端口名称，默认为 "x"
        fallback_first: 当端口名不匹配时，是否返回第一个值
    
    返回:
        torch.Tensor 或 None
    
    示例:
        >>> extract_single_input({"x": tensor}, "x")  # 返回 tensor
        >>> extract_single_input(tensor, "x")          # 直接返回 tensor
        >>> extract_single_input({"y": tensor}, "x")   # fallback_first=True 时返回 tensor
    """
    # 空值快速路径
    if inputs is None:
        return None
    
    # 已经是张量，直接返回
    if isinstance(inputs, torch.Tensor):
        return inputs
    
    # 字典类型：按端口名或第一个值提取
    if isinstance(inputs, dict):
        if port_name in inputs:
            return _to_tensor(inputs[port_name])
        if fallback_first and len(inputs) > 0:
            return _to_tensor(next(iter(inputs.values())))
        return None
    
    # 列表类型：尝试转换为张量
    if isinstance(inputs, (list, tuple)):
        return _to_tensor(inputs)
    
    # 其他类型：尝试转换
    return _to_tensor(inputs)


def extract_multi_input(
    inputs: Any, 
    port_names: List[str],
    *,
    strict: bool = False
) -> Dict[str, Optional[torch.Tensor]]:
    """
    从输入中提取多个张量
    
    参数:
        inputs: dict 类型的输入
        port_names: 端口名称列表，如 ["x", "y"]
        strict: 严格模式，缺少端口时返回空字典
    
    返回:
        dict 包含各端口的张量
    
    示例:
        >>> extract_multi_input({"x": t1, "y": t2}, ["x", "y"])
        {"x": t1, "y": t2}
    """
    # 空输入处理
    if inputs is None:
        return {name: None for name in port_names}
    
    # 非字典类型：将其作为第一个端口的值
    if not isinstance(inputs, dict):
        result = {name: None for name in port_names}
        if port_names:
            result[port_names[0]] = _to_tensor(inputs)
        return result
    
    # 严格模式检查
    if strict:
        for name in port_names:
            if name not in inputs:
                return {}
    
    # 按端口名提取
    return {name: _to_tensor(inputs.get(name)) for name in port_names}


def ensure_tensor(
    value: Any, 
    dtype: torch.dtype = torch.float32,
    *,
    device: Optional[str] = None
) -> Optional[torch.Tensor]:
    """
    确保值为张量格式
    
    参数:
        value: 任意值（list, number, tensor 等）
        dtype: 目标数据类型
        device: 目标设备（如 "cuda:0"）
    
    返回:
        torch.Tensor 或 None
    """
    if value is None:
        return None
    
    if isinstance(value, torch.Tensor):
        tensor = value.to(dtype) if value.dtype != dtype else value
        return tensor.to(device) if device else tensor
    
    try:
        tensor = torch.tensor(value, dtype=dtype)
        return tensor.to(device) if device else tensor
    except (ValueError, TypeError, RuntimeError):
        return None


def get_shape(tensor: Any) -> Optional[List[int]]:
    """
    安全获取张量形状
    
    参数:
        tensor: torch.Tensor 或其他对象
    
    返回:
        list 形状列表，或 None
    """
    if tensor is None:
        return None
    
    if hasattr(tensor, 'shape'):
        return list(tensor.shape)
    
    if isinstance(tensor, (list, tuple)):
        return _infer_list_shape(tensor)
    
    return None


def broadcast_shapes(*shapes: ShapeLike) -> Optional[List[int]]:
    """
    计算多个形状广播后的结果形状
    
    参数:
        *shapes: 多个形状列表
    
    返回:
        广播后的形状，或 None（不可广播时）
    
    示例:
        >>> broadcast_shapes([3, 1], [1, 4])
        [3, 4]
    """
    # 过滤空值
    valid_shapes = [list(s) for s in shapes if s is not None]
    if not valid_shapes:
        return None
    
    try:
        # 使用PyTorch内置的广播逻辑
        dummy_tensors = [torch.empty(s) for s in valid_shapes]
        result = torch.broadcast_shapes(*[t.shape for t in dummy_tensors])
        return list(result)
    except RuntimeError:
        return None


# ==================== 内部辅助函数 ====================

def _to_tensor(value: Any) -> Optional[torch.Tensor]:
    """内部函数：将值转换为张量"""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value
    try:
        return torch.tensor(value)
    except (ValueError, TypeError, RuntimeError):
        return None


def _infer_list_shape(lst: Union[list, tuple]) -> List[int]:
    """内部函数：推断嵌套列表的形状"""
    shape = []
    current = lst
    while isinstance(current, (list, tuple)):
        shape.append(len(current))
        if len(current) == 0:
            break
        current = current[0]
    return shape
