"""
registry.py - 节点注册表

用法：
    import registry
    registry.registerCategory(id, label, color, icon)  # 注册分类
    registry.registerNode(opcode, label, category, ports, params, func)  # 注册节点
    registry.getAllForFrontend()  # 获取前端格式的节点数据
    
示例：
    registry.registerCategory("math", "数学运算", "#FF6B6B", "base64...")
    registry.registerNode("add", "加法", "math", {"in": ["a", "b"], "out": ["result"]}, {}, myFunc)
    data = registry.getAllForFrontend()  # 返回{categories: [...], nodes: [...]}
"""

nodes = {}  # 全局变量：节点定义字典，key是opcode，value是节点信息
categories = {}  # 全局变量：分类定义字典，key是id，value是分类信息


def registerCategory(id, label, color, icon):
    """
    注册分类
    
    用法：
        registerCategory("math", "数学运算", "#FF6B6B", "base64...")
        
    示例：
        registerCategory("layers", "神经网络层", "#4ECDC4", "data:image/png;base64,...")
        registerCategory("activations", "激活函数", "#FFE66D", "")
    """
    cat = {}  # 创建空字典准备装分类信息
    cat["id"] = id  # 分类唯一标识
    cat["label"] = label  # 分类显示名称
    cat["color"] = color  # 分类颜色，用于前端显示
    cat["icon"] = icon  # 分类图标，base64格式
    categories[id] = cat  # 存入全局categories字典


def registerNode(opcode, label, category, ports, params, func):
    """
    注册节点
    
    用法：
        registerNode("add", "加法", "math", {"in": ["a", "b"], "out": ["result"]}, {}, addFunc)
        
    示例：
        registerNode("linear", "线性层", "layers", {"in": ["x"], "out": ["y"]}, {"units": 64}, linearFunc)
        registerNode("relu", "ReLU", "activations", {"in": ["x"], "out": ["y"]}, {}, reluFunc)
    """
    node = {}  # 创建空字典准备装节点信息
    node["opcode"] = opcode  # 节点操作码，唯一标识
    node["label"] = label  # 节点显示名称
    node["category"] = category  # 节点所属分类的id
    node["ports"] = ports  # 节点的输入输出端口定义
    node["params"] = params  # 节点的参数定义
    node["func"] = func  # 节点的执行函数（包含infer、build、compute）
    nodes[opcode] = node  # 存入全局nodes字典


def getAllForFrontend():
    """
    获取前端格式数据
    
    用法：
        data = getAllForFrontend()
        
    示例：
        result = getAllForFrontend()  # 返回{categories: [...], nodes: [...]}
        # categories是分类数组，nodes是去掉func的节点数组
    """
    catList = []  # 创建空数组准备装分类列表
    for catId in categories:  # 遍历所有分类
        catList.append(categories[catId])  # 把每个分类加入数组
    
    nodeList = []  # 创建空数组准备装节点列表
    for opcode in nodes:  # 遍历所有节点
        node = nodes[opcode]  # 取出节点信息
        nodeForFrontend = {}  # 创建新字典，不包含func
        nodeForFrontend["opcode"] = node["opcode"]  # 复制opcode
        nodeForFrontend["label"] = node["label"]  # 复制label
        nodeForFrontend["category"] = node["category"]  # 复制category
        nodeForFrontend["ports"] = node["ports"]  # 复制ports
        nodeForFrontend["params"] = node["params"]  # 复制params
        nodeList.append(nodeForFrontend)  # 把节点加入数组，不包含func
    
    result = {}  # 创建结果字典
    result["categories"] = catList  # 分类列表
    result["nodes"] = nodeList  # 节点列表（不含func）
    return result  # 返回前端格式的数据
