CATEGORIES = {}   
NODES = {}        
CURRENT_CATEGORY = None  

def category(id, name, color, icon):
    """类别装饰器"""
    def decorator(func):
        global CURRENT_CATEGORY
        CATEGORIES[id] = {
            "id": id,
            "name": name,
            "color": color,
            "icon": icon
        }
        CURRENT_CATEGORY = id 
        return func
    return decorator

def node(opcode, name, ports, params):
    """节点装饰器"""
    def decorator(func):
        global CURRENT_CATEGORY
        if CURRENT_CATEGORY is None:
            raise ValueError(f"节点 {opcode} 没有定义在任何类别下！先用 @category 定义类别。")
        
        NODES[opcode] = {
            "opcode": opcode,
            "name": name,
            "ports": ports,
            "params": params,
            "func": func,
            "category": CURRENT_CATEGORY 
        }
        return func
    return decorator