"""
容错处理工具模块

提供安全调用、自动重试、异常捕获等功能。
设计原则：让代码更加健壮，无论遇到什么异常都能优雅处理。
"""

import functools
import time
import traceback
from typing import Any, Callable, Dict, Optional, TypeVar, Union

T = TypeVar('T')
R = TypeVar('R')

# ==================== 安全调用 ====================

def safe_call(
    func: Callable[..., R],
    *args,
    default: T = None,
    on_error: Optional[Callable[[Exception], Any]] = None,
    **kwargs
) -> Union[R, T]:
    """
    安全调用函数，自动捕获异常
    
    参数:
        func: 要调用的函数
        *args: 位置参数
        default: 发生异常时的默认返回值
        on_error: 异常处理回调函数
        **kwargs: 关键字参数
    
    返回:
        函数返回值，或 default
    
    示例:
        >>> safe_call(risky_function, arg1, arg2, default=0)
        >>> safe_call(parse_int, "abc", default=0)
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if on_error:
            on_error(e)
        return default


def safe_call_async(
    default: T = None,
    on_error: Optional[Callable[[Exception], Any]] = None
) -> Callable:
    """
    异步函数的安全调用装饰器
    
    参数:
        default: 发生异常时的默认返回值
        on_error: 异常处理回调函数
    
    返回:
        装饰器
    
    示例:
        @safe_call_async(default={})
        async def risky_async_func():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if on_error:
                    on_error(e)
                return default
        return wrapper
    return decorator


# ==================== 安全取值 ====================

def safe_get(
    obj: Any,
    *keys,
    default: T = None
) -> Union[Any, T]:
    """
    安全地从嵌套对象中获取值
    
    参数:
        obj: 对象（dict, list, 或任意对象）
        *keys: 键路径
        default: 获取失败时的默认值
    
    返回:
        获取的值，或 default
    
    示例:
        >>> safe_get({"a": {"b": 1}}, "a", "b")  # 1
        >>> safe_get({"a": 1}, "b", "c", default=0)  # 0
        >>> safe_get([1, [2, 3]], 1, 0)  # 2
    """
    current = obj
    for key in keys:
        try:
            if isinstance(current, dict):
                current = current.get(key)
            elif isinstance(current, (list, tuple)):
                current = current[key] if isinstance(key, int) and 0 <= key < len(current) else None
            elif hasattr(current, key):
                current = getattr(current, key)
            else:
                return default
            
            if current is None:
                return default
        except (KeyError, IndexError, TypeError, AttributeError):
            return default
    
    return current if current is not None else default


def with_default(value: Any, default: T) -> Union[Any, T]:
    """
    如果值为空则返回默认值
    
    参数:
        value: 待检查的值
        default: 默认值
    
    返回:
        value 或 default
    
    示例:
        >>> with_default(None, 42)  # 42
        >>> with_default("", "default")  # "default"
        >>> with_default(0, 42)  # 0 (0 不是空值)
    """
    if value is None:
        return default
    if isinstance(value, str) and value == "":
        return default
    if isinstance(value, (list, dict)) and len(value) == 0:
        return default
    return value


def first_valid(*values, default: T = None) -> Union[Any, T]:
    """
    返回第一个非空的值
    
    参数:
        *values: 多个值
        default: 所有值都为空时的默认值
    
    返回:
        第一个非空值，或 default
    
    示例:
        >>> first_valid(None, "", [], 42, "hello")  # 42
    """
    for value in values:
        if value is not None:
            if isinstance(value, str) and value == "":
                continue
            if isinstance(value, (list, dict)) and len(value) == 0:
                continue
            return value
    return default


# ==================== 异常处理 ====================

def catch_and_log(
    *,
    message: str = "操作失败",
    default: T = None,
    reraise: bool = False,
    log_func: Callable = print
) -> Callable:
    """
    捕获异常并记录日志的装饰器
    
    参数:
        message: 错误消息前缀
        default: 发生异常时的默认返回值
        reraise: 是否重新抛出异常
        log_func: 日志函数
    
    返回:
        装饰器
    
    示例:
        @catch_and_log(message="解析失败", default={})
        def parse_data(data):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log_func(f"⚠️ {message}: {e}")
                if reraise:
                    raise
                return default
        return wrapper
    return decorator


def catch_and_log_async(
    *,
    message: str = "操作失败",
    default: T = None,
    reraise: bool = False,
    log_func: Callable = print
) -> Callable:
    """
    异步版本的异常捕获装饰器
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                log_func(f"⚠️ {message}: {e}")
                if reraise:
                    raise
                return default
        return wrapper
    return decorator


def get_error_info(e: Exception) -> Dict[str, Any]:
    """
    获取异常的详细信息
    
    参数:
        e: 异常对象
    
    返回:
        包含异常信息的字典
    """
    return {
        "type": type(e).__name__,
        "message": str(e),
        "traceback": traceback.format_exc(),
    }


# ==================== 重试机制 ====================

def retry(
    max_attempts: int = 3,
    delay: float = 0.1,
    *,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    default: T = None
) -> Callable:
    """
    自动重试装饰器
    
    参数:
        max_attempts: 最大重试次数
        delay: 重试间隔（秒）
        exceptions: 需要重试的异常类型
        on_retry: 重试时的回调函数
        default: 所有重试都失败后的默认返回值
    
    返回:
        装饰器
    
    示例:
        @retry(max_attempts=3, delay=0.5)
        def unstable_operation():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if on_retry:
                        on_retry(attempt, e)
                    if attempt < max_attempts:
                        time.sleep(delay)
            
            if default is not None:
                return default
            if last_exception:
                raise last_exception
            return None
        return wrapper
    return decorator


def retry_async(
    max_attempts: int = 3,
    delay: float = 0.1,
    *,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    default: T = None
) -> Callable:
    """
    异步版本的自动重试装饰器
    """
    import asyncio
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if on_retry:
                        on_retry(attempt, e)
                    if attempt < max_attempts:
                        await asyncio.sleep(delay)
            
            if default is not None:
                return default
            if last_exception:
                raise last_exception
            return None
        return wrapper
    return decorator


# ==================== 条件执行 ====================

def run_if(
    condition: Union[bool, Callable[[], bool]],
    func: Callable[..., R],
    *args,
    default: T = None,
    **kwargs
) -> Union[R, T]:
    """
    条件满足时才执行函数
    
    参数:
        condition: 条件（布尔值或返回布尔值的函数）
        func: 要执行的函数
        *args: 位置参数
        default: 条件不满足时的返回值
        **kwargs: 关键字参数
    
    返回:
        函数返回值，或 default
    """
    should_run = condition() if callable(condition) else condition
    if should_run:
        return func(*args, **kwargs)
    return default


def guard(
    check: Callable[[Any], bool],
    *,
    default: T = None,
    transform: Optional[Callable[[Any], Any]] = None
) -> Callable:
    """
    参数守卫装饰器
    
    在函数执行前检查第一个参数是否满足条件。
    
    参数:
        check: 检查函数
        default: 检查失败时的默认返回值
        transform: 检查通过后对参数的转换函数
    
    返回:
        装饰器
    
    示例:
        @guard(lambda x: x is not None, default=0)
        def process(value):
            return value * 2
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(first_arg, *args, **kwargs):
            if not check(first_arg):
                return default
            if transform:
                first_arg = transform(first_arg)
            return func(first_arg, *args, **kwargs)
        return wrapper
    return decorator
