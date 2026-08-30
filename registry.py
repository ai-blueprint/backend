import threading  # 注册表同步能力，用于插件重载和蓝图编译互斥
import torch.nn as nn

nodes = {}
categories = {}
registryLock = threading.RLock()  # 编译和插件重载不能观察半成品注册表
nodeOwners = {}  # 记录每个操作码的归属，插件不能静默覆盖内置节点
categoryOwners = {}  # 记录每个分类的归属，重载结果可明确追踪
registrationOwner = "core"  # 动态导入期间由插件加载器临时切换所有者
currentCategoryId = None  # 保存当前节点文件正在使用的分类上下文

friendlyLabels = {
    "relu": "负数变零", "sigmoid": "压到0和1", "tanh": "压到负1和1", "softmax": "变成概率", "softplus": "平滑变正", "leakyRelu": "负数保留一点", "elu": "负数平滑变换", "gelu": "平滑门控",
    "linear": "线性层", "conv": "卷积层", "pooling": "池化层", "dropout": "随机丢弃", "embedding": "数字变向量", "layer_norm": "按特征归一化", "group_norm": "分组归一化", "batch_norm": "按批归一化", "instance_norm": "按样本归一化", "rms_norm": "均方根归一化",
    "reshape": "改变形状", "transpose": "交换两个维度", "permute": "重排列维度", "squeeze": "去掉单维度", "unsqueeze": "增加单维度", "flatten": "压平成一维", "unflatten": "拆开一个维度", "pad": "边缘补值", "detach": "切断梯度", "clone": "复制张量", "slice": "截取一段", "select": "选择一个位置", "cat": "连接张量", "stack": "叠起张量", "expand": "扩大尺寸", "time_shift": "向前移位",
    "add": "相加", "sub": "相减", "mul": "相乘", "div": "相除", "matmul": "矩阵相乘", "bmm": "批量矩阵相乘", "einsum": "按公式计算", "lerp": "线性插值", "dot": "点积", "pow": "幂运算", "norm": "计算长度", "exp": "指数", "sqrt": "平方根", "sum": "求和", "abs": "取绝对值", "neg": "取相反数", "mean": "求平均值",
    "mse_loss": "平均平方误差", "cross_entropy_loss": "分类误差", "l1_loss": "绝对值误差", "bce_loss": "二分类误差",
}

categoriesOrder = ["base", "transform", "activation", "attention", "loss", "normalization", "shape", "math"]


def clearAll():  # 清空注册表，热重载时调用
    global currentCategoryId
    nodes.clear()  # 清空节点字典
    categories.clear()  # 清空分类字典
    nodeOwners.clear()  # 清空节点归属，等待重新加载建立真实所有权
    categoryOwners.clear()  # 清空分类归属，避免保留失效插件记录
    currentCategoryId = None  # 清空分类上下文，等待下一个节点文件重新声明


def setRegistrationOwner(owner):
    global registrationOwner
    registrationOwner = owner or "core"  # 空所有者统一回退到内置节点身份


def registerCategory(id, label, color, icon):
    global currentCategoryId
    existingOwner = categoryOwners.get(id)  # 读取已注册分类的归属
    if existingOwner and existingOwner != registrationOwner:
        raise ValueError(f"分类冲突: {id} 已由 {existingOwner} 注册")  # 不允许插件覆盖其他所有者分类
    existingNodes = categories.get(id, {}).get("nodes", [])  # 同一分类可能由多个节点文件共同声明
    categories[id] = {"label": label, "color": color, "icon": icon, "nodes": existingNodes}  # 更新分类资料但保留已注册节点
    categoryOwners[id] = registrationOwner  # 分类创建成功后记录当前导入所有者
    currentCategoryId = id  # 即使分类已存在，也把当前节点文件切换到它


def registerNode(opcode, label, ports, params, description, cls):
    existingOwner = nodeOwners.get(opcode)  # 读取已注册操作码的归属
    if existingOwner and existingOwner != registrationOwner:
        raise ValueError(f"节点冲突: {opcode} 已由 {existingOwner} 注册")  # 冲突必须显式失败而不是替换实现
    if not currentCategoryId or currentCategoryId not in categories:
        raise ValueError(f"节点 {opcode} 注册前必须先声明分类")  # 隐式使用最后分类需要先有明确分类
    nodes[opcode] = {"opcode": opcode, "label": friendlyLabels.get(opcode, label), "ports": ports, "params": params, "description": description, "cls": cls}
    categories[currentCategoryId]["nodes"].append(opcode)  # 节点归入当前分类上下文，不依赖字典顺序
    nodeOwners[opcode] = registrationOwner  # 节点创建成功后记录当前导入所有者


def getAllForFrontend():
    # 根据categoriesOrder排序，categoriesOrder是优先级顺序，不是所有分类都要包含
    result = {
        "categories": {},
        "nodes": {},
    }  # 初始化返回结果字典，包含分类和节点两个部分

    hiddenCategories = {"example_category"}  # 前端默认隐藏的分类集合

    # 第一步：生成优先级分类列表
    priority = [(catId, categories[catId]) for catId in categoriesOrder if catId in categories and catId not in hiddenCategories]  # 按优先级顺序提取存在且非隐藏的分类

    # 第二步：生成剩余分类列表
    used = set(catId for catId, _ in priority)  # 提取已使用的分类ID集合，用于后续过滤
    remaining = [(catId, cat) for catId, cat in categories.items() if catId not in used and catId not in hiddenCategories]  # 提取未使用且非隐藏的分类

    # 第三步：拼接列表并转换成字典
    for catId, cat in priority + remaining:  # 遍历合并后的分类列表，先优先级再剩余
        result["categories"][catId] = {k: v for k, v in cat.items() if k != "cls"}  # 添加分类信息到结果，过滤掉cls属性

    # 第四步：收集所有可见节点的opcode
    visibleOpcodes = set()  # 可见节点opcode集合
    for catId, cat in result["categories"].items():  # 遍历可见分类
        visibleOpcodes.update(cat.get("nodes", []))  # 收集该分类下的所有节点opcode

    # 第五步：构建节点数据（仅包含可见分类下的节点）
    for opcode, node in nodes.items():  # 遍历所有节点
        if opcode not in visibleOpcodes:
            continue  # 跳过隐藏分类下的节点
        result["nodes"][opcode] = {k: v for k, v in node.items() if k != "cls"}  # 添加节点信息到结果，过滤掉cls属性

    return result  # 返回排序后的结果


def validateParams(opcode, params):
    """根据注册定义做通用参数校验，统一返回扁平值字典。"""
    definition = nodes[opcode].get("params", {})  # 获取节点参数定义
    if not isinstance(params, dict):
        params = {}  # 入参不是字典时使用空字典兜底

    validated = {}  # 存储校验后的扁平参数

    for key, spec in definition.items():  # 先遍历定义，保证缺失参数也有默认值
        defaultValue = spec.get("value")  # 默认值
        value = params.get(key, defaultValue)  # 前端传值优先，没有就用默认值

        if isinstance(value, dict) and "value" in value:
            value = value.get("value")  # 兼容旧格式参数对象

        if value is None:
            value = defaultValue  # 空值回退默认值

        options = spec.get("options")  # 选项限制
        if options and value not in options:
            print(f"参数选项无效：{opcode}.{key}={value}，回退默认值{defaultValue}")  # 打印修正日志
            value = defaultValue  # 非法选项回退默认值

        paramRange = spec.get("range")  # 范围限制
        canClamp = isinstance(value, (int, float)) and not isinstance(value, bool)  # 只对数值做范围修正
        hasRange = isinstance(paramRange, (list, tuple)) and len(paramRange) == 2  # 范围配置必须是两个边界值
        if canClamp and hasRange:
            minValue = paramRange[0]  # 最小边界
            maxValue = paramRange[1]  # 最大边界
            corrected = max(minValue, min(maxValue, value))  # 执行夹逼修正
            if corrected != value:
                print(f"参数越界修正：{opcode}.{key}={value}，修正为{corrected}")  # 打印修正日志
            value = corrected  # 写入修正结果

        validated[key] = value  # 保存当前参数结果

    for key, value in params.items():  # 保留定义外的扩展参数，保证参数适配宽容
        if key in validated:
            continue  # 已处理过的键直接跳过
        if isinstance(value, dict) and "value" in value:
            value = value.get("value")  # 兼容旧格式扩展参数
        validated[key] = value  # 原样保留扩展参数

    return validated  # 返回扁平参数字典


def createNode(opcode, nodeId, params):
    """根据opcode创建节点实例，创建前校验参数"""
    if opcode not in nodes:
        raise ValueError(f"未知节点: {opcode}")
    params = validateParams(opcode, params)  # 校验并修正参数值
    cls = nodes[opcode]["cls"]
    return cls(nodeId, params)


def category(id="", label="", color="#8992eb", icon=""):
    registerCategory(id, label, color, icon)


def node(opcode="", label="", ports={}, params={}, description=""):
    def decorator(cls):
        registerNode(opcode, label, ports, params, description, cls)
        return cls

    return decorator


class BaseNode(nn.Module):
    def __init__(self, nodeId, params):
        super().__init__()
        self.nodeId = nodeId
        self.params = params
        self.build()

    def build(self):
        pass

    def compute(self, input):
        raise NotImplementedError("必须实现compute")

    def forward(self, input):
        out = self.compute(input)
        # 占位，到时候做值存储和转发操作
        return out
