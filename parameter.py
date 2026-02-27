from abc import abstractmethod, ABC
from collections.abc import Generator
from typing import Type, Any

from util import putIntoList

ValidateResult = tuple[bool, Any]
RangeTuple = tuple[float, float]
TypeGroup = tuple[Type, ...] | Type


def formatTypeList(typeGroup: TypeGroup) -> str:
    return ", ".join(t.__name__ for t in putIntoList(typeGroup))


def errorMessage(error_type: str, msg: str):
    return f"参数{error_type}：{msg}"


def typeError(
    opcode: str,
    key: str,
    expected_type: TypeGroup,
    conflict_type: TypeGroup,
    actual_value: Any,
    defaultValue: Any,
):
    return errorMessage(
        "类型错误",
        f"{opcode}.{key} 期望{formatTypeList(expected_type)}，但不接受{formatTypeList(conflict_type)}，实际{type(actual_value).__name__}={actual_value}，回退默认值{defaultValue}",
    )


def boundaryCurrection(
    opcode: str,
    key: str,
    value: Any,
    expected_range: RangeTuple,
    corrected_value: Any,
):
    return errorMessage(
        "越界修正",
        f"{opcode}.{key} 值{value}超出范围{expected_range}，修正为{corrected_value}",
    )


def invalidOption(
    opcode: str,
    key: str,
    value: Any,
    valid_options: tuple[Any, ...],
    default: Any,
):
    return errorMessage(
        "选项无效",
        f"{opcode}.{key} 值{value}不在选项{valid_options}中，回退默认值{default}",
    )


class ParameterTypeBase(ABC):
    def __init__(
        self,
        typeName: str,
        baseType: TypeGroup = object,
        conflictType: TypeGroup = object,
    ) -> None:
        self.baseType = baseType
        self.conflictType = conflictType
        self.typeName = typeName

    @abstractmethod
    def _customValidate(
        self,
        value: Any,
        opcode: str,
        key: str,
        defaultValue: Any,
        paramRange: RangeTuple | None = None,
        paramOptions: dict | None = None,
    ) -> Generator[ValidateResult]:
        pass

    def expectRange(
        self,
        value: Any,
        opcode: str,
        key: str,
        paramRange: RangeTuple | None = None,
    ) -> Generator[ValidateResult]:
        if paramRange and len(paramRange) == 2:
            clamped = max(paramRange[0], min(paramRange[1], value))
            if clamped != value:
                yield (
                    False,
                    boundaryCurrection(opcode, key, value, paramRange, clamped),
                )
            yield True, clamped

    def expectType(
        self,
        value: Any,
        opcode: str,
        key: str,
        defaultValue: Any,
    ) -> Generator[ValidateResult]:
        if not isinstance(value, self.baseType) or isinstance(value, self.conflictType):
            yield (
                False,
                typeError(
                    opcode,
                    key,
                    self.baseType,
                    self.conflictType,
                    value,
                    defaultValue,
                ),
            )
            yield True, defaultValue
        else:
            yield True, value

    def expectOptions(
        self,
        value: Any,
        opcode: str,
        key: str,
        paramOptions: dict | None = None,
        defaultValue: Any = None,
    ) -> Generator[ValidateResult]:
        if paramOptions and value not in paramOptions:
            yield (
                False,
                invalidOption(
                    opcode,
                    key,
                    value,
                    tuple(paramOptions.keys()),
                    defaultValue,
                ),
            )
            yield True, defaultValue
        else:
            yield True, value

    def validate(
        self,
        value: Any,
        opcode: str,
        key: str,
        defaultValue: Any,
        paramRange: RangeTuple | None = None,
        paramOptions: dict | None = None,
    ) -> ValidateResult:
        if isinstance(value, dict) and "value" in value:
            value = value["value"]
        for valid, dataOrMessage in self._customValidate(
            value,
            opcode,
            key,
            defaultValue,
            paramRange,
            paramOptions,
        ):
            if valid:
                value = dataOrMessage
            else:
                print(dataOrMessage)
        return True, value
