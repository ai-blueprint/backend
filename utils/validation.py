"""
参数验证工具模块

提供参数类型检查、范围验证、类型强制转换等功能。
设计原则：验证失败时尽可能返回合理的默认值，而非抛出异常。
"""

from typing import Any, Dict, List, Optional, Union, Callable, TypeVar

T = TypeVar('T')

# ==================== 类型验证 ====================

def validate_type(
    value: Any,
    expected_type: type,
    *,
    default: T = None,
    allow_none: bool = True
) -> Union[Any, T]:
    """
    验证值的类型
    
    参数:
        value: 待验证的值
        expected_type: 期望的类型
        default: 验证失败时的默认值
        allow_none: 是否允许 None 值
    
    返回:
        通过验证的值，或默认值
    
    示例:
        >>> validate_type(42, int)  # 42
        >>> validate_type("hello", int, default=0)  # 0
    """
    if value is None:
        return None if allow_none else default
    
    if isinstance(value, expected_type):
        return value
    
    return default


def validate_types(
    value: Any,
    expected_types: tuple,
    *,
    default: T = None,
    allow_none: bool = True
) -> Union[Any, T]:
    """
    验证值是否为多个类型之一
    
    参数:
        value: 待验证的值
        expected_types: 期望类型的元组，如 (int, float)
        default: 验证失败时的默认值
        allow_none: 是否允许 None 值
    
    返回:
        通过验证的值，或默认值
    """
    if value is None:
        return None if allow_none else default
    
    if isinstance(value, expected_types):
        return value
    
    return default


# ==================== 范围验证 ====================

def validate_range(
    value: Any,
    *,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    default: T = None,
    clamp: bool = False
) -> Union[float, int, T]:
    """
    验证数值是否在指定范围内
    
    参数:
        value: 待验证的数值
        min_val: 最小值（包含）
        max_val: 最大值（包含）
        default: 验证失败时的默认值
        clamp: 是否将超出范围的值截断到边界
    
    返回:
        通过验证的值，截断后的值，或默认值
    
    示例:
        >>> validate_range(5, min_val=0, max_val=10)  # 5
        >>> validate_range(15, min_val=0, max_val=10, clamp=True)  # 10
        >>> validate_range(15, min_val=0, max_val=10)  # None
    """
    if value is None:
        return default
    
    # 尝试转换为数值
    try:
        num_value = float(value)
    except (ValueError, TypeError):
        return default
    
    # 检查范围
    if min_val is not None and num_value < min_val:
        if clamp:
            return type(value)(min_val) if isinstance(value, int) else min_val
        return default
    
    if max_val is not None and num_value > max_val:
        if clamp:
            return type(value)(max_val) if isinstance(value, int) else max_val
        return default
    
    return value


def validate_in(
    value: Any,
    valid_values: List[Any],
    *,
    default: T = None,
    case_insensitive: bool = False
) -> Union[Any, T]:
    """
    验证值是否在允许的列表中
    
    参数:
        value: 待验证的值
        valid_values: 允许的值列表
        default: 验证失败时的默认值
        case_insensitive: 对字符串进行大小写不敏感比较
    
    返回:
        通过验证的值，或默认值
    """
    if value is None:
        return default
    
    if case_insensitive and isinstance(value, str):
        lower_value = value.lower()
        for v in valid_values:
            if isinstance(v, str) and v.lower() == lower_value:
                return v  # 返回原始列表中的值
        return default
    
    if value in valid_values:
        return value
    
    return default


# ==================== 类型强制转换 ====================

def coerce_type(
    value: Any,
    target_type: str,
    *,
    default: T = None
) -> Union[Any, T]:
    """
    强制将值转换为目标类型
    
    参数:
        value: 待转换的值
        target_type: 目标类型名称 ("number", "int", "float", "string", "boolean")
        default: 转换失败时的默认值
    
    返回:
        转换后的值，或默认值
    
    示例:
        >>> coerce_type("42", "number")  # 42
        >>> coerce_type("true", "boolean")  # True
        >>> coerce_type(123, "string")  # "123"
    """
    if value is None:
        return default
    
    # 空字符串特殊处理
    if value == '' or value == "":
        return default
    
    try:
        if target_type == "number":
            return _to_number(value)
        elif target_type == "int":
            return int(float(value))
        elif target_type == "float":
            return float(value)
        elif target_type == "string":
            return str(value)
        elif target_type == "boolean":
            return _to_boolean(value)
        else:
            return value
    except (ValueError, TypeError):
        return default


def coerce_params(
    params: Dict[str, Any],
    schema: Dict[str, str],
    *,
    strict: bool = False
) -> Dict[str, Any]:
    """
    根据schema批量转换参数类型
    
    参数:
        params: 参数字典
        schema: 类型定义 {param_name: type_name}
        strict: 严格模式，转换失败时移除该参数
    
    返回:
        转换后的参数字典
    
    示例:
        >>> coerce_params({"lr": "0.01", "epochs": "100"}, {"lr": "float", "epochs": "int"})
        {"lr": 0.01, "epochs": 100}
    """
    result = {}
    for key, value in params.items():
        if key in schema:
            coerced = coerce_type(value, schema[key])
            if coerced is not None or not strict:
                result[key] = coerced
        else:
            result[key] = value
    return result


# ==================== 参数规范化 ====================

def normalize_params(
    raw_params: Dict[str, Any],
    *,
    extract_defaults: bool = True
) -> Dict[str, Any]:
    """
    规范化参数格式
    
    处理两种格式：
    1. 旧格式：{"param": value}
    2. 新格式：{"param": {"label": "...", "type": "...", "default": value}}
    
    参数:
        raw_params: 原始参数字典
        extract_defaults: 是否从新格式中提取 default 值
    
    返回:
        规范化后的参数字典 {"param": value}
    """
    if not raw_params:
        return {}
    
    result = {}
    for key, value in raw_params.items():
        if isinstance(value, dict) and extract_defaults:
            # 新格式：提取 default 值
            if 'default' in value:
                default_val = value['default']
                param_type = value.get('type', 'string')
                result[key] = coerce_type(default_val, param_type, default=default_val)
            else:
                result[key] = value
        else:
            # 旧格式：直接使用
            result[key] = value
    
    return result


def validate_params(
    params: Dict[str, Any],
    schema: Dict[str, Dict[str, Any]],
    *,
    fill_defaults: bool = True
) -> Dict[str, Any]:
    """
    验证参数字典
    
    schema格式：
    {
        "param_name": {
            "type": "number|string|boolean",
            "required": True/False,
            "default": any_value,
            "min": number,
            "max": number,
            "choices": [...]
        }
    }
    
    参数:
        params: 待验证的参数
        schema: 参数定义schema
        fill_defaults: 是否填充缺失参数的默认值
    
    返回:
        验证并处理后的参数字典
    """
    result = {}
    
    for name, rules in schema.items():
        value = params.get(name)
        default = rules.get('default')
        
        # 必填检查
        if value is None:
            if rules.get('required', False):
                continue  # 跳过必填但缺失的参数
            if fill_defaults and default is not None:
                result[name] = default
            continue
        
        # 类型转换
        param_type = rules.get('type')
        if param_type:
            value = coerce_type(value, param_type, default=default)
        
        # 范围验证
        if param_type in ('number', 'int', 'float'):
            value = validate_range(
                value,
                min_val=rules.get('min'),
                max_val=rules.get('max'),
                default=default,
                clamp=rules.get('clamp', False)
            )
        
        # 枚举验证
        if 'choices' in rules:
            value = validate_in(value, rules['choices'], default=default)
        
        if value is not None:
            result[name] = value
    
    # 保留schema中未定义的参数
    for name, value in params.items():
        if name not in result and name not in schema:
            result[name] = value
    
    return result


# ==================== 内部辅助函数 ====================

def _to_number(value: Any) -> Union[int, float]:
    """内部函数：将值转换为数值"""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    
    str_value = str(value).strip()
    if '.' in str_value or 'e' in str_value.lower():
        return float(str_value)
    return int(str_value)


def _to_boolean(value: Any) -> bool:
    """内部函数：将值转换为布尔值"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on', 'y')
    return bool(value)
