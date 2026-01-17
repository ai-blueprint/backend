"""
工具函数模块：提供节点计算和引擎执行的通用辅助函数

设计原则：
1. 所有函数都要处理边界情况
2. 输入格式自动适配（字典/张量/None）
3. 提供清晰的错误信息
"""

import torch                                                                    # 导入PyTorch库


def extract_single_input(inputs, port_name="x"):                                # 从输入中提取单个张量
    """
    从输入中提取单个张量，智能处理多种输入格式
    
    参数：
        inputs: 可以是 dict、Tensor 或 None
        port_name: 端口名称，默认为 "x"
    
    返回：
        torch.Tensor 或 None
    """
    if inputs is None:                                                          # 空输入
        return None                                                             # 直接返回None
    
    if isinstance(inputs, torch.Tensor):                                        # 如果已经是张量
        return inputs                                                           # 直接返回
    
    if isinstance(inputs, dict):                                                # 如果是字典
        if port_name in inputs:                                                 # 优先按端口名取
            return inputs[port_name]                                            # 返回对应端口的值
        if len(inputs) > 0:                                                     # 如果有值但端口名不匹配
            return list(inputs.values())[0]                                     # 返回第一个值
        return None                                                             # 空字典返回None
    
    return inputs                                                               # 其他情况直接返回


def extract_multi_input(inputs, port_names):                                    # 从输入中提取多个张量
    """
    从输入中提取多个张量
    
    参数：
        inputs: dict 类型的输入
        port_names: 端口名称列表，如 ["x", "y"]
    
    返回：
        dict 包含各端口的张量
    """
    if inputs is None:                                                          # 空输入
        return {name: None for name in port_names}                              # 返回全None字典
    
    if not isinstance(inputs, dict):                                            # 如果不是字典
        return {port_names[0]: inputs} if port_names else {}                    # 将其作为第一个端口的值
    
    return {name: inputs.get(name) for name in port_names}                      # 按端口名提取


def safe_tensor_operation(func, *args, default=None, **kwargs):                 # 安全执行张量操作
    """
    安全执行张量操作，自动处理异常
    
    参数：
        func: 要执行的函数
        args: 位置参数
        default: 发生异常时的默认返回值
        kwargs: 关键字参数
    
    返回：
        func的返回值，或 default
    """
    try:                                                                        # 尝试执行
        return func(*args, **kwargs)                                            # 调用函数
    except Exception as e:                                                      # 捕获异常
        print(f"⚠️ 张量操作异常: {e}")                                            # 打印警告
        return default                                                          # 返回默认值


def ensure_tensor(value, dtype=torch.float32):                                  # 确保值为张量格式
    """
    确保值为张量格式
    
    参数：
        value: 任意值（list, number, tensor 等）
        dtype: 目标数据类型
    
    返回：
        torch.Tensor
    """
    if value is None:                                                           # 空值
        return None                                                             # 返回None
    
    if isinstance(value, torch.Tensor):                                         # 已经是张量
        return value                                                            # 直接返回
    
    try:                                                                        # 尝试转换
        return torch.tensor(value, dtype=dtype)                                 # 转换为张量
    except Exception:                                                           # 转换失败
        return None                                                             # 返回None


def get_shape(tensor):                                                          # 获取张量形状
    """
    安全获取张量形状
    
    参数：
        tensor: torch.Tensor 或 None
    
    返回：
        list 形状列表，或 None
    """
    if tensor is None:                                                          # 空值
        return None                                                             # 返回None
    
    if hasattr(tensor, 'shape'):                                                # 如果有shape属性
        return list(tensor.shape)                                               # 返回形状列表
    
    return None                                                                 # 无法获取形状


def serialize_tensor(tensor):                                                   # 序列化张量为JSON格式
    """
    将张量序列化为可JSON化的格式
    
    参数：
        tensor: torch.Tensor
    
    返回：
        dict 包含类型、形状和数据
    """
    if tensor is None:                                                          # 空值
        return None                                                             # 返回None
    
    if not isinstance(tensor, torch.Tensor):                                    # 不是张量
        return tensor                                                           # 直接返回原值
    
    return {                                                                    # 构造序列化结果
        "type": "tensor",                                                       # 类型标识
        "shape": list(tensor.shape),                                            # 张量形状
        "data": tensor.tolist()                                                 # 张量数据
    }


def deserialize_tensor(data, dtype=torch.float32):                              # 反序列化JSON为张量
    """
    将JSON格式反序列化为张量
    
    参数：
        data: dict 或 list
        dtype: 目标数据类型
    
    返回：
        torch.Tensor
    """
    if data is None:                                                            # 空值
        return None                                                             # 返回None
    
    if isinstance(data, torch.Tensor):                                          # 已经是张量
        return data                                                             # 直接返回
    
    if isinstance(data, dict) and data.get("type") == "tensor":                 # 序列化格式
        return torch.tensor(data["data"], dtype=dtype)                          # 从data字段恢复
    
    if isinstance(data, list):                                                  # 列表格式
        return torch.tensor(data, dtype=dtype)                                  # 直接转换
    
    return None                                                                 # 无法转换
