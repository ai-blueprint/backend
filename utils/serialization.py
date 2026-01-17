"""
序列化工具模块

提供张量和数据结构的序列化/反序列化功能。
设计目标：无论输入什么格式，都能安全地序列化为JSON兼容格式。
"""

import torch
from typing import Any, Dict, List, Optional, Union

# ==================== 类型别名 ====================
JSONValue = Union[Dict, List, str, int, float, bool, None]


def serialize_tensor(tensor: Any) -> Optional[Dict[str, Any]]:
    """
    将张量序列化为JSON兼容的字典格式
    
    参数:
        tensor: torch.Tensor 或其他类型
    
    返回:
        {"type": "tensor", "shape": [...], "data": [...]} 或 None
    
    示例:
        >>> serialize_tensor(torch.tensor([1, 2, 3]))
        {"type": "tensor", "shape": [3], "data": [1, 2, 3]}
    """
    if tensor is None:
        return None
    
    if not isinstance(tensor, torch.Tensor):
        return None  # 非张量返回 None，让调用者处理
    
    try:
        return {
            "type": "tensor",
            "shape": list(tensor.shape),
            "data": tensor.detach().cpu().tolist(),
            "dtype": str(tensor.dtype).split('.')[-1],  # 如 "float32"
        }
    except (RuntimeError, TypeError):
        return None


def deserialize_tensor(
    data: Any, 
    dtype: torch.dtype = torch.float32,
    *,
    device: Optional[str] = None
) -> Optional[torch.Tensor]:
    """
    将JSON格式反序列化为张量
    
    支持多种输入格式：
    1. {"type": "tensor", "data": [...]} - 标准序列化格式
    2. [...] - 纯列表格式
    3. torch.Tensor - 直接返回
    
    参数:
        data: dict, list 或 Tensor
        dtype: 目标数据类型
        device: 目标设备
    
    返回:
        torch.Tensor 或 None
    """
    if data is None:
        return None
    
    # 已经是张量
    if isinstance(data, torch.Tensor):
        return _apply_dtype_device(data, dtype, device)
    
    # 标准序列化格式
    if isinstance(data, dict) and data.get("type") == "tensor":
        tensor_data = data.get("data")
        if tensor_data is None:
            return None
        try:
            tensor = torch.tensor(tensor_data, dtype=dtype)
            return _apply_dtype_device(tensor, dtype, device)
        except (ValueError, TypeError, RuntimeError):
            return None
    
    # 纯列表/元组格式
    if isinstance(data, (list, tuple)):
        try:
            tensor = torch.tensor(data, dtype=dtype)
            return _apply_dtype_device(tensor, dtype, device)
        except (ValueError, TypeError, RuntimeError):
            return None
    
    # 标量数值
    if isinstance(data, (int, float)):
        try:
            tensor = torch.tensor(data, dtype=dtype)
            return _apply_dtype_device(tensor, dtype, device)
        except (ValueError, TypeError):
            return None
    
    return None


def serialize_value(value: Any) -> JSONValue:
    """
    将单个值转换为JSON兼容格式
    
    处理规则：
    1. torch.Tensor -> {"type": "tensor", ...}
    2. dict -> 递归序列化
    3. list -> 递归序列化
    4. 基本类型 -> 直接返回
    
    参数:
        value: 任意值
    
    返回:
        JSON兼容的值
    """
    # None
    if value is None:
        return None
    
    # 张量
    if isinstance(value, torch.Tensor):
        return serialize_tensor(value)
    
    # 字典：递归处理
    if isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    
    # 列表/元组：递归处理
    if isinstance(value, (list, tuple)):
        # 检查是否是嵌套的数值列表（可能是张量数据）
        if _is_numeric_nested_list(value):
            return list(value)  # 保持原样
        return [serialize_value(item) for item in value]
    
    # numpy数组
    if hasattr(value, 'tolist'):
        return value.tolist()
    
    # 基本类型
    if isinstance(value, (str, int, float, bool)):
        return value
    
    # 其他类型：尝试转换为字符串
    try:
        return str(value)
    except Exception:
        return None


def serialize_output(output: Any) -> JSONValue:
    """
    序列化节点输出结果
    
    专门用于序列化引擎执行后的节点输出。
    
    参数:
        output: 节点输出（通常是 dict 或 Tensor）
    
    返回:
        JSON兼容的输出
    """
    if output is None:
        return None
    
    if isinstance(output, dict):
        return {key: serialize_value(val) for key, val in output.items()}
    
    return serialize_value(output)


def serialize_all_outputs(results: Dict[str, Any]) -> Dict[str, JSONValue]:
    """
    批量序列化所有节点的输出
    
    参数:
        results: {node_id: output, ...}
    
    返回:
        {node_id: serialized_output, ...}
    """
    if results is None:
        return {}
    
    return {
        node_id: serialize_output(output) 
        for node_id, output in results.items()
    }


# ==================== JSON工具函数 ====================

def to_json_safe(obj: Any) -> JSONValue:
    """
    将任意对象转换为JSON安全格式
    
    这是一个更激进的转换函数，会尽一切可能转换为JSON兼容格式。
    """
    return serialize_value(obj)


def from_json(
    data: JSONValue, 
    *,
    auto_tensor: bool = True,
    dtype: torch.dtype = torch.float32
) -> Any:
    """
    从JSON格式恢复Python对象
    
    参数:
        data: JSON数据
        auto_tensor: 是否自动将序列化格式恢复为张量
        dtype: 张量的目标数据类型
    
    返回:
        恢复后的Python对象
    """
    if data is None:
        return None
    
    # 检查是否是张量序列化格式
    if auto_tensor and isinstance(data, dict) and data.get("type") == "tensor":
        return deserialize_tensor(data, dtype)
    
    # 递归处理字典
    if isinstance(data, dict):
        return {k: from_json(v, auto_tensor=auto_tensor, dtype=dtype) for k, v in data.items()}
    
    # 递归处理列表
    if isinstance(data, list):
        return [from_json(item, auto_tensor=auto_tensor, dtype=dtype) for item in data]
    
    # 其他类型直接返回
    return data


# ==================== 内部辅助函数 ====================

def _apply_dtype_device(
    tensor: torch.Tensor, 
    dtype: Optional[torch.dtype] = None,
    device: Optional[str] = None
) -> torch.Tensor:
    """内部函数：应用数据类型和设备"""
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def _is_numeric_nested_list(value: Any) -> bool:
    """内部函数：检查是否是纯数值的嵌套列表"""
    if not isinstance(value, (list, tuple)):
        return False
    
    if len(value) == 0:
        return True
    
    first = value[0]
    if isinstance(first, (int, float)):
        return all(isinstance(x, (int, float)) for x in value)
    if isinstance(first, (list, tuple)):
        return all(_is_numeric_nested_list(x) for x in value)
    
    return False
