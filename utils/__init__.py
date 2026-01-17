"""
工具函数包：提供项目通用的辅助功能

模块结构：
- tensor: 张量操作工具（提取、转换、形状处理）
- serialization: 序列化工具（张量转JSON、反序列化）
- validation: 参数验证工具（类型检查、范围验证）
- safe: 容错处理工具（安全调用、自动重试）

使用方式：
    from utils import extract_input, serialize, validate_params
    # 或
    from utils.tensor import extract_single_input
"""

# ==================== 张量工具 ====================
from utils.tensor import (
    extract_single_input,      # 从输入中提取单个张量
    extract_multi_input,       # 从输入中提取多个张量
    ensure_tensor,             # 确保值为张量格式
    get_shape,                 # 安全获取张量形状
    broadcast_shapes,          # 计算广播后的形状
)

# ==================== 序列化工具 ====================
from utils.serialization import (
    serialize_tensor,          # 张量 -> JSON格式
    deserialize_tensor,        # JSON格式 -> 张量
    serialize_value,           # 通用值序列化
    serialize_output,          # 输出结果序列化
    serialize_all_outputs,     # 批量序列化所有输出
)

# ==================== 验证工具 ====================
from utils.validation import (
    validate_params,           # 验证参数字典
    validate_type,             # 验证值类型
    validate_range,            # 验证数值范围
    coerce_type,               # 强制类型转换
    normalize_params,          # 规范化参数格式
)

# ==================== 容错工具 ====================
from utils.safe import (
    safe_call,                 # 安全调用函数
    safe_get,                  # 安全获取字典值
    with_default,              # 带默认值的取值
    catch_and_log,             # 捕获异常并记录
    retry,                     # 自动重试装饰器
)

# ==================== 图算法工具 ====================
from utils.graph import (
    topological_sort,          # 拓扑排序
    get_node_inputs,           # 获取节点输入
    get_downstream_nodes,      # 获取下游节点
    get_upstream_nodes,        # 获取上游节点
)

# ==================== 版本信息 ====================
__version__ = "1.0.0"
__all__ = [
    # 张量工具
    "extract_single_input", "extract_multi_input", "ensure_tensor",
    "get_shape", "broadcast_shapes",
    # 序列化工具
    "serialize_tensor", "deserialize_tensor", "serialize_value",
    "serialize_output", "serialize_all_outputs",
    # 验证工具
    "validate_params", "validate_type", "validate_range",
    "coerce_type", "normalize_params",
    # 容错工具
    "safe_call", "safe_get", "with_default", "catch_and_log", "retry",
    # 图算法工具
    "topological_sort", "get_node_inputs", "get_downstream_nodes", "get_upstream_nodes",
]
