"""
expressions.py - 蓝图变量与参数表达式解析

变量来自蓝图variables列表，节点参数输入框既可以填普通数值，也可以填B、D*4或[B, S, D]。
解析器只支持数字、变量名、加减乘除、整除、括号和逗号，逐字符解析，不执行任何代码。
用法：value = resolveText("D*4", {"D": 8})
"""

import re  # 变量名校验能力，用于拒绝非法标识符


variableNamePattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")  # 变量名只允许字母数字下划线且不能以数字开头


# --- 把浮点整值收敛为整数 ---
def normalizeNumber(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)  # D/HEADS这类整除结果可以直接用于层宽等整数参数
    return value  # 非整浮点保留原值，是否合法交给具体节点判断


# --- 把表达式文本切成词法单元 ---
def tokenize(text):
    tokens = []  # 顺序保存(类型, 值)词法单元
    index = 0  # 当前扫描位置
    while index < len(text):
        char = text[index]  # 读取当前字符
        if char.isspace():
            index += 1  # 空白只用于分隔，直接跳过
            continue
        if char.isdigit() or (char == "." and index + 1 < len(text) and text[index + 1].isdigit()):
            start = index  # 记录数字起点
            while index < len(text) and (text[index].isdigit() or text[index] == "."):
                index += 1  # 连续吃掉整数和小数部分
            tokens.append(("number", float(text[start:index])))  # 统一按浮点读取，求值后再收敛整数
            continue
        if char.isalpha() or char == "_":
            start = index  # 记录变量名起点
            while index < len(text) and (text[index].isalnum() or text[index] == "_"):
                index += 1  # 连续吃掉字母数字下划线
            tokens.append(("name", text[start:index]))  # 变量名在求值阶段查表
            continue
        if text.startswith("//", index):
            tokens.append(("op", "//"))  # 整除必须先于普通除号匹配
            index += 2
            continue
        if char in "+-*/(),":
            tokens.append(("op", char))  # 单字符运算符直接入列
            index += 1
            continue
        raise ValueError(f"表达式包含无法识别的字符: {char}")  # 其他字符一律拒绝，杜绝任何代码注入
    return tokens  # 返回可被递归下降解析的词法序列


class ExpressionParser:
    """按加减、乘除整除、一元负号和括号的标准优先级即时求值。"""

    def __init__(self, tokens, variables):
        self.tokens = tokens  # 待解析词法单元
        self.position = 0  # 当前解析位置
        self.variables = variables  # 变量名到数值的映射

    # --- 查看当前词法单元 ---
    def peek(self):
        return self.tokens[self.position] if self.position < len(self.tokens) else (None, None)  # 越界返回空单元表示结束

    # --- 按需消费一个运算符 ---
    def match(self, *operators):
        kind, value = self.peek()  # 读取但不移动
        if kind == "op" and value in operators:
            self.position += 1  # 命中后才前进
            return value
        return None  # 未命中保持位置不变

    # --- 校验参与算术的值是数字 ---
    def requireNumber(self, value, operator):
        if isinstance(value, list):
            raise ValueError(f"列表变量不能参与{operator}运算")  # 形状列表只能整体引用，不能加减乘除
        return value  # 数字原样返回给运算

    # --- 解析逗号分隔的顶层序列 ---
    def parseSequence(self):
        values = [self.parseExpression()]  # 序列至少包含一个表达式
        while self.match(","):
            values.append(self.parseExpression())  # 逗号后继续读取下一项
        return values  # 返回顶层各项求值结果

    # --- 解析加减层 ---
    def parseExpression(self):
        value = self.parseTerm()  # 先取乘除层结果作为左值
        while True:
            operator = self.match("+", "-")
            if not operator:
                return value  # 没有后续加减时返回当前值
            left = self.requireNumber(value, operator)  # 左值必须是数字
            right = self.requireNumber(self.parseTerm(), operator)  # 右值同样必须是数字
            value = left + right if operator == "+" else left - right  # 立即求值保持单遍解析

    # --- 解析乘除层 ---
    def parseTerm(self):
        value = self.parseUnary()  # 先取一元层结果作为左值
        while True:
            operator = self.match("*", "//", "/")
            if not operator:
                return value  # 没有后续乘除时返回当前值
            left = self.requireNumber(value, operator)  # 左值必须是数字
            right = self.requireNumber(self.parseUnary(), operator)  # 右值同样必须是数字
            if operator in ("/", "//") and right == 0:
                raise ValueError("表达式出现除以零")  # 除零直接失败并给出明确原因
            if operator == "*":
                value = left * right  # 乘法
            elif operator == "/":
                value = left / right  # 真除法，整值结果稍后收敛为整数
            else:
                value = left // right  # 整除

    # --- 解析一元负号 ---
    def parseUnary(self):
        if self.match("-"):
            return -self.requireNumber(self.parseUnary(), "负号")  # 负号只作用于数字
        return self.parseAtom()  # 无负号时读取基本单元

    # --- 解析数字、变量和括号 ---
    def parseAtom(self):
        kind, value = self.peek()  # 读取当前单元
        if kind == "number":
            self.position += 1  # 消费数字
            return value
        if kind == "name":
            self.position += 1  # 消费变量名
            if value not in self.variables:
                raise ValueError(f"未定义的变量: {value}")  # 未定义引用必须显式失败
            return self.variables[value]  # 变量值可以是数字或形状列表
        if self.match("("):
            inner = self.parseExpression()  # 括号内是完整表达式
            if not self.match(")"):
                raise ValueError("表达式缺少右括号")  # 括号必须成对
            return inner
        raise ValueError("表达式结构不完整")  # 空位置或非法开头统一报错


# --- 解析并求值一段表达式文本 ---
def resolveText(text, variables):
    """
    用法：resolveText("B, S, D*2", {"B": 2, "S": 4, "D": 8}) 返回 [2, 4, 16]
    单个表达式返回数字，逗号序列或列表变量返回数字列表。
    """
    stripped = str(text).strip()  # 前后空白不影响语义
    if stripped.startswith("[") and stripped.endswith("]"):
        stripped = stripped[1:-1]  # 允许用户带方括号书写形状
    if not stripped.strip():
        raise ValueError("表达式不能为空")  # 空内容无法产生参数值

    parser = ExpressionParser(tokenize(stripped), variables)  # 构建单遍求值解析器
    values = parser.parseSequence()  # 解析顶层逗号序列
    if parser.position != len(parser.tokens):
        raise ValueError("表达式存在多余内容")  # 未消费的词法单元说明语法错误

    flattened = []  # 顶层序列中出现的列表变量按元素展开
    for value in values:
        if isinstance(value, list):
            flattened.extend(value)  # SHAPE这类列表变量并入序列
        else:
            flattened.append(value)  # 数字直接保留

    normalized = [normalizeNumber(item) for item in flattened]  # 所有整值浮点统一收敛为整数
    if len(values) == 1 and not isinstance(values[0], list):
        return normalized[0]  # 单个数字表达式返回标量
    return normalized  # 序列和列表变量统一返回数字列表


# --- 校验并构建变量名到值的映射 ---
def getVariablesMap(variableList):
    """
    用法：getVariablesMap([{"id": "v1", "name": "D", "value": 8}]) 返回 {"D": 8}
    """
    variables = {}  # 保存已通过校验的变量
    for item in variableList or []:
        name = str(item.get("name", "")) if isinstance(item, dict) else ""  # 非对象条目按无名处理
        if not variableNamePattern.match(name):
            raise ValueError(f"变量名无效: {name}")  # 名称必须是合法标识符才能进入表达式
        if name in variables:
            raise ValueError(f"变量名重复: {name}")  # 重名会让引用产生歧义

        value = item.get("value")  # 变量值只允许具体数字或数字列表
        if isinstance(value, bool) or not isinstance(value, (int, float, list)):
            raise ValueError(f"变量{name}的值必须是数字或数字列表")  # 布尔和其他类型不能参与形状表达
        if isinstance(value, list):
            if not value or not all(isinstance(entry, (int, float)) and not isinstance(entry, bool) for entry in value):
                raise ValueError(f"变量{name}的列表值必须全部是数字")  # 空列表和混合类型都无法作为形状
            value = [normalizeNumber(entry) for entry in value]  # 列表内整值浮点同样收敛
        else:
            value = normalizeNumber(value)  # 标量整值浮点收敛为整数
        variables[name] = value  # 记录合法变量
    return variables  # 返回表达式求值可直接使用的映射


# --- 递归解析单个参数值 ---
def resolveValue(value, variables):
    if isinstance(value, dict) and "expr" in value:
        return resolveText(value.get("expr", ""), variables)  # 表达式对象求值为具体数字或列表
    if isinstance(value, list):
        resolved = []  # 列表内也允许混入表达式对象
        for item in value:
            itemValue = resolveValue(item, variables)  # 逐项解析
            if isinstance(itemValue, list):
                resolved.extend(itemValue)  # 列表结果按元素并入外层
            else:
                resolved.append(itemValue)  # 普通值直接保留
        return resolved
    return value  # 数字、布尔和字符串原样通过


# --- 解析节点全部参数为具体值 ---
def resolveNodeParams(params, variables):
    """
    用法：resolveNodeParams({"out_shape": {"value": {"expr": "B, S, D"}}}, {"B": 2, "S": 4, "D": 8})
    返回扁平参数字典 {"out_shape": [2, 4, 8]}，可直接交给registry.createNode。
    """
    resolved = {}  # 输出扁平参数，节点创建不再感知表达式
    for key, spec in (params or {}).items():
        value = spec.get("value") if isinstance(spec, dict) and "value" in spec else spec  # 兼容参数对象和扁平值
        resolved[key] = resolveValue(value, variables)  # 表达式在此处彻底消解
    return resolved  # 返回可直接校验和构建的参数
