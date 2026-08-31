import re  # 操作码命名校验能力，保证节点身份格式统一
import threading  # 注册表同步能力，用于插件重载和蓝图编译互斥
import torch.nn as nn

nodes = {}
categories = {}
registryLock = threading.RLock()  # 编译和插件重载不能观察半成品注册表
nodeOwners = {}  # 记录每个操作码的归属，插件不能静默覆盖内置节点
categoryOwners = {}  # 记录每个分类的归属，重载结果可明确追踪
registrationOwner = "core"  # 动态导入期间由插件加载器临时切换所有者
currentCategoryId = None  # 保存当前节点文件正在使用的分类上下文
nodeAliases = {
    "leakyRelu": ("leaky_relu", {}),  # 兼容旧版驼峰操作码，新的蓝图统一使用下划线命名
    "dropout1d": ("dropout", {"mode": "channel1d"}), "dropout2d": ("dropout", {"mode": "channel2d"}), "dropout3d": ("dropout", {"mode": "channel3d"}), "alpha_dropout": ("dropout", {"mode": "alpha"}), "feature_alpha_dropout": ("dropout", {"mode": "feature_alpha"}),
    "zeros_like": ("tensor_like", {"mode": "zeros"}), "ones_like": ("tensor_like", {"mode": "ones"}), "rand_like": ("tensor_like", {"mode": "random"}),
    "floor": ("rounding", {"mode": "floor"}), "ceil": ("rounding", {"mode": "ceil"}), "round": ("rounding", {"mode": "round"}), "trunc": ("rounding", {"mode": "trunc"}),
    "pixel_shuffle": ("pixel_rearrange", {"mode": "up", "factorFrom": "upscale_factor"}), "pixel_unshuffle": ("pixel_rearrange", {"mode": "down", "factorFrom": "downscale_factor"}),
}  # 旧蓝图操作码集中迁移到统一节点

technicalLabels = {
    "input": "输入", "output": "输出", "debug": "调试", "linear": "Linear", "conv": "Conv", "pooling": "池化", "dropout": "Dropout", "embedding": "Embedding", "bilinear": "Bilinear", "conv_transpose": "ConvTranspose", "upsample": "Upsample", "channel_shuffle": "ChannelShuffle", "pixel_shuffle": "PixelShuffle", "pixel_unshuffle": "PixelUnshuffle", "identity": "Identity", "layer_norm": "LayerNorm", "group_norm": "GroupNorm", "batch_norm": "BatchNorm", "instance_norm": "InstanceNorm", "rms_norm": "RMSNorm",
    "relu": "ReLU", "sigmoid": "Sigmoid", "tanh": "Tanh", "softmax": "Softmax", "softplus": "Softplus", "leaky_relu": "LeakyReLU", "elu": "ELU", "gelu": "GELU", "relu6": "ReLU6", "rrelu": "RReLU", "selu": "SELU", "celu": "CELU", "silu": "SiLU", "mish": "Mish", "hardsigmoid": "Hardsigmoid", "hardswish": "Hardswish", "log_sigmoid": "LogSigmoid", "prelu": "PReLU", "hardshrink": "Hardshrink", "softshrink": "Softshrink", "tanhshrink": "Tanhshrink", "threshold": "Threshold", "glu": "GLU", "softmin": "Softmin", "log_softmax": "LogSoftmax",
    "multihead_attention": "MultiheadAttention", "scaled_dot_product_attention": "scaled_dot_product_attention", "cross_attention": "交叉注意力", "mse_loss": "MSELoss", "cross_entropy_loss": "CrossEntropyLoss", "l1_loss": "L1Loss", "bce_loss": "BCEWithLogitsLoss", "smooth_l1_loss": "SmoothL1Loss", "huber_loss": "HuberLoss", "poisson_nll_loss": "PoissonNLLLoss", "gaussian_nll_loss": "GaussianNLLLoss", "kl_div_loss": "KLDivLoss", "margin_ranking_loss": "MarginRankingLoss", "triplet_margin_loss": "TripletMarginLoss", "cosine_embedding_loss": "CosineEmbeddingLoss",
    "reshape": "reshape", "transpose": "transpose", "permute": "permute", "squeeze": "squeeze", "unsqueeze": "unsqueeze", "flatten": "flatten", "unflatten": "unflatten", "pad": "pad", "detach": "detach", "clone": "clone", "slice": "切片", "select": "select", "cat": "cat", "stack": "stack", "expand": "expand", "time_shift": "时间移位", "chunk": "chunk", "roll": "roll", "flip": "flip", "repeat_interleave": "repeat_interleave",
    "add": "add", "sub": "sub", "mul": "mul", "div": "div", "matmul": "matmul", "bmm": "bmm", "einsum": "einsum", "lerp": "lerp", "dot": "dot", "pow": "pow", "norm": "norm", "exp": "exp", "sqrt": "sqrt", "sum": "sum", "abs": "abs", "neg": "neg", "mean": "mean", "log": "log", "log10": "log10", "log2": "log2", "log1p": "log1p", "expm1": "expm1", "exp2": "exp2", "square": "square", "signbit": "signbit", "trunc": "trunc", "maximum": "maximum", "minimum": "minimum", "remainder": "remainder", "fmod": "fmod", "hypot": "hypot", "clamp": "clamp", "sign": "sign", "floor": "floor", "ceil": "ceil", "round": "round", "frac": "frac", "reciprocal": "reciprocal", "rsqrt": "rsqrt", "sin": "sin", "cos": "cos", "tan": "tan", "atan": "atan", "sinh": "sinh", "cosh": "cosh", "erf": "erf", "amax": "amax", "amin": "amin", "prod": "prod", "var": "var", "std": "std", "argmax": "argmax", "greater": "gt", "greater_equal": "ge", "less": "lt", "less_equal": "le", "equal": "eq", "zeros_like": "zeros_like", "ones_like": "ones_like", "rand_like": "rand_like", "max_pool1d": "MaxPool1d", "avg_pool1d": "AvgPool1d", "adaptive_avg_pool1d": "AdaptiveAvgPool1d", "dropout1d": "Dropout1d", "dropout2d": "Dropout2d", "dropout3d": "Dropout3d", "alpha_dropout": "AlphaDropout", "feature_alpha_dropout": "FeatureAlphaDropout", "local_response_norm": "LocalResponseNorm", "reflection_pad1d": "ReflectionPad1d", "cosine_similarity": "CosineSimilarity", "pairwise_distance": "PairwiseDistance", "scan": "cumsum", "fold": "cumprod", "cumulative_max": "cummax", "cumulative_min": "cummin",
}
friendlyLabels = technicalLabels  # 保留旧变量名，避免外部插件读取技术名称时发生兼容性断裂

chineseLabels = {
    "relu": "负数变零", "sigmoid": "压到零和一", "tanh": "压到负一和一", "softmax": "变成概率", "softplus": "平滑变正", "leaky_relu": "泄漏ReLU", "elu": "负数平滑变化", "gelu": "平滑门控", "relu6": "限制在零到六", "rrelu": "随机泄漏", "selu": "自归一化", "celu": "连续指数", "silu": "平滑门控", "mish": "平滑自门控", "hardsigmoid": "快速Sigmoid", "hardswish": "快速Swish", "log_sigmoid": "对数Sigmoid", "prelu": "参数化ReLU", "hardshrink": "硬收缩", "softshrink": "软收缩", "tanhshrink": "Tanh收缩", "threshold": "阈值替换", "glu": "门控单元", "softmin": "Softmin", "log_softmax": "对数Softmax",
    "linear": "线性层", "conv": "卷积层", "pooling": "池化", "dropout": "随机丢弃", "embedding": "嵌入", "bilinear": "双输入线性层", "conv_transpose": "转置卷积", "upsample": "上采样", "channel_shuffle": "通道重排", "pixel_shuffle": "像素上采样", "pixel_unshuffle": "像素下采样", "reflection_pad1d": "反射填充", "local_response_norm": "局部归一化", "dropout1d": "一维随机丢弃", "alpha_dropout": "Alpha随机丢弃", "feature_alpha_dropout": "特征随机丢弃", "cosine_similarity": "余弦相似度", "pairwise_distance": "成对距离", "max_pool1d": "一维最大池化", "avg_pool1d": "一维平均池化", "adaptive_avg_pool1d": "一维自适应平均池化", "dropout2d": "二维随机丢弃", "dropout3d": "三维随机丢弃",
    "multihead_attention": "多头注意力", "scaled_dot_product_attention": "缩放点积注意力", "cross_attention": "交叉注意力", "mse_loss": "平均平方误差", "cross_entropy_loss": "分类误差", "l1_loss": "绝对值误差", "bce_loss": "二分类误差", "smooth_l1_loss": "平滑绝对值误差", "huber_loss": "Huber误差", "poisson_nll_loss": "泊松误差", "gaussian_nll_loss": "高斯误差", "kl_div_loss": "分布差异", "margin_ranking_loss": "排序误差", "triplet_margin_loss": "三元组误差", "cosine_embedding_loss": "方向相似误差",
    "layer_norm": "层归一化", "group_norm": "组归一化", "batch_norm": "批归一化", "instance_norm": "样本归一化", "rms_norm": "均方根归一化", "reshape": "改变形状", "transpose": "交换维度", "permute": "重排维度", "squeeze": "去掉单维度", "unsqueeze": "增加单维度", "flatten": "压平维度", "unflatten": "拆开维度", "pad": "填充边缘", "select": "选择位置", "cat": "连接张量", "stack": "堆叠张量", "expand": "扩大尺寸", "time_shift": "时间移位", "chunk": "分成几块", "roll": "循环移位", "flip": "翻转顺序", "repeat_interleave": "重复元素",
    "add": "相加", "sub": "相减", "mul": "相乘", "div": "相除", "matmul": "矩阵相乘", "bmm": "批量矩阵相乘", "einsum": "按公式运算", "lerp": "线性插值", "dot": "点积", "pow": "幂运算", "norm": "计算长度", "exp": "指数", "sqrt": "平方根", "sum": "求和", "abs": "绝对值", "neg": "相反数", "mean": "平均值", "log": "自然对数", "log10": "常用对数", "log2": "二进制对数", "log1p": "加一取对数", "expm1": "减一取指数", "exp2": "二的指数", "square": "平方", "signbit": "判断负数", "trunc": "截断小数", "maximum": "逐个取较大值", "minimum": "逐个取较小值", "remainder": "取余数", "fmod": "浮点取余", "hypot": "直角距离", "clamp": "限制范围", "sign": "取符号", "floor": "向下取整", "ceil": "向上取整", "round": "四舍五入", "frac": "取小数", "reciprocal": "取倒数", "rsqrt": "倒数平方根", "sin": "正弦", "cos": "余弦", "tan": "正切", "atan": "反正切", "sinh": "双曲正弦", "cosh": "双曲余弦", "erf": "误差函数", "amax": "最大值", "amin": "最小值", "prod": "乘积", "var": "方差", "std": "标准差", "argmax": "最大值位置", "greater": "大于", "greater_equal": "大于等于", "less": "小于", "less_equal": "小于等于", "equal": "相等", "zeros_like": "同形全零", "ones_like": "同形全一", "rand_like": "同形随机", "input": "输入", "output": "输出", "debug": "调试",
}

categoriesOrder = ["base", "transform", "activation", "loss", "normalization", "shape", "math"]
opcodePattern = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")  # 新操作码统一使用小写下划线格式


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
    if not isinstance(opcode, str) or not opcodePattern.fullmatch(opcode):
        raise ValueError(f"节点操作码必须使用snake_case命名: {opcode}")  # 阻止新的驼峰或大写操作码进入注册表
    existingOwner = nodeOwners.get(opcode)  # 读取已注册操作码的归属
    if existingOwner and existingOwner != registrationOwner:
        raise ValueError(f"节点冲突: {opcode} 已由 {existingOwner} 注册")  # 冲突必须显式失败而不是替换实现
    if not currentCategoryId or currentCategoryId not in categories:
        raise ValueError(f"节点 {opcode} 注册前必须先声明分类")  # 隐式使用最后分类需要先有明确分类
    nodes[opcode] = {"opcode": opcode, "technicalLabel": technicalLabels.get(opcode, opcode), "label": chineseLabels.get(opcode, label), "ports": ports, "params": params, "description": description, "cls": cls}
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


def resolveNodeAlias(opcode, params):
    if opcode not in nodeAliases:
        return opcode, params  # 当前操作码直接使用原定义
    targetOpcode, aliasParams = nodeAliases[opcode]  # 读取统一目标和固定模式
    resolvedParams = dict(params or {})  # 复制旧参数避免修改蓝图数据
    factorFrom = aliasParams.get("factorFrom")  # 像素重排旧节点的倍数参数名不同
    if factorFrom:
        resolvedParams["factor"] = resolvedParams.get(factorFrom, 2)  # 迁移旧倍数参数
    resolvedParams.update({key: value for key, value in aliasParams.items() if key != "factorFrom"})  # 写入固定模式
    return targetOpcode, resolvedParams  # 返回统一节点和迁移参数


def hasNode(opcode):
    return opcode in nodes or opcode in nodeAliases  # 新节点和旧蓝图别名都可执行


def createNode(opcode, nodeId, params):
    """根据opcode创建节点实例，创建前校验参数"""
    if not hasNode(opcode):
        raise ValueError(f"未知节点: {opcode}")
    opcode, params = resolveNodeAlias(opcode, params)  # 旧蓝图先迁移到统一节点
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
