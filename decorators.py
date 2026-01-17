"""
节点注册装饰器模块

提供 @category 和 @node 装饰器，用于声明式地注册节点。

使用示例:
    @category(id="math", name="数学运算", color="#FFB6C1", icon="...")
    def math_category():
        pass
    
    @node(opcode="add", name="加法", ports={"in": ["x", "y"], "out": ["result"]}, params={})
    def add_node():
        def infer(input_shapes, params): ...
        def build(params): ...
        def compute(inputs, layer): ...
        return infer, build, compute
"""

from typing import Any, Callable, Dict, List, Optional

# ==================== 全局存储 ====================

CATEGORIES: Dict[str, Dict[str, Any]] = {}  # 分类注册表
NODES: Dict[str, Dict[str, Any]] = {}       # 节点注册表
CURRENT_CATEGORY: Optional[str] = None       # 当前激活的分类


# ==================== 装饰器 ====================

def category(
    id: str,
    name: str,
    color: str = "#888888",
    icon: str = ""
) -> Callable:
    """
    分类装饰器
    
    用于定义一个节点分类，后续的 @node 装饰器会自动归入此分类。
    
    参数:
        id: 分类唯一标识符
        name: 分类显示名称
        color: 分类主题颜色（十六进制）
        icon: 分类图标（base64或URL）
    
    返回:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        global CURRENT_CATEGORY
        
        # 验证参数
        if not id or not isinstance(id, str):
            raise ValueError("分类 id 必须是非空字符串")
        if not name or not isinstance(name, str):
            raise ValueError("分类 name 必须是非空字符串")
        
        # 注册分类
        CATEGORIES[id] = {
            "id": id,
            "name": name,
            "color": color,
            "icon": icon
        }
        
        # 设置当前分类
        CURRENT_CATEGORY = id
        
        return func
    
    return decorator


def node(
    opcode: str,
    name: str,
    ports: Optional[Dict[str, List[str]]] = None,
    params: Optional[Dict[str, Any]] = None
) -> Callable:
    """
    节点装饰器
    
    用于注册一个节点定义。节点函数必须返回 (infer, build, compute) 三元组。
    
    参数:
        opcode: 节点唯一标识符（操作码）
        name: 节点显示名称
        ports: 端口定义 {"in": [...], "out": [...]}
        params: 参数定义 {"param_name": default_value}
    
    返回:
        装饰器函数
    
    示例:
        @node(
            opcode="linear",
            name="全连接层",
            ports={"in": ["x"], "out": ["out"]},
            params={"in_features": 128, "out_features": 64}
        )
        def linear_layer():
            def infer(input_shapes, params): ...
            def build(input_shapes, params): ...
            def compute(x, layer): ...
            return infer, build, compute
    """
    def decorator(func: Callable) -> Callable:
        global CURRENT_CATEGORY
        
        # 验证参数
        if not opcode or not isinstance(opcode, str):
            raise ValueError("节点 opcode 必须是非空字符串")
        if not name or not isinstance(name, str):
            raise ValueError("节点 name 必须是非空字符串")
        
        # 检查是否在分类下
        if CURRENT_CATEGORY is None:
            raise ValueError(
                f"节点 {opcode} 没有定义在任何分类下！"
                f"请先用 @category 定义分类。"
            )
        
        # 规范化端口定义
        normalized_ports = _normalize_ports(ports)
        
        # 规范化参数定义
        normalized_params = params if params is not None else {}
        
        # 注册节点
        NODES[opcode] = {
            "opcode": opcode,
            "name": name,
            "ports": normalized_ports,
            "params": normalized_params,
            "func": func,
            "category": CURRENT_CATEGORY
        }
        
        return func
    
    return decorator


# ==================== 辅助函数 ====================

def _normalize_ports(ports: Optional[Dict]) -> Dict[str, List[str]]:
    """规范化端口定义"""
    if ports is None:
        return {"in": [], "out": []}
    
    return {
        "in": ports.get("in", []),
        "out": ports.get("out", [])
    }


def reset_registry():
    """重置所有注册表（用于测试或重新加载）"""
    global CURRENT_CATEGORY
    CATEGORIES.clear()
    NODES.clear()
    CURRENT_CATEGORY = None


def get_node(opcode: str) -> Optional[Dict]:
    """获取节点定义"""
    return NODES.get(opcode)


def get_category(cat_id: str) -> Optional[Dict]:
    """获取分类定义"""
    return CATEGORIES.get(cat_id)


def set_current_category(cat_id: str):
    """手动设置当前分类（用于测试或特殊场景）"""
    global CURRENT_CATEGORY
    if cat_id not in CATEGORIES:
        raise ValueError(f"分类 {cat_id} 不存在")
    CURRENT_CATEGORY = cat_id
