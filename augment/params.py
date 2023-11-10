import abc
import dataclasses
import numbers
import typing


@dataclasses.dataclass
class Param(abc.ABC):
    name: str


NT = typing.TypeVar("NT", bound=numbers.Number)


@dataclasses.dataclass
class NumberParam(typing.Generic[NT], Param):
    min_value: typing.Optional[NT]
    max_value: typing.Optional[NT]


class IntegerParam(NumberParam[int]):
    pass


class FloatParam(NumberParam[float]):
    pass


CT = typing.TypeVar("CT")


@dataclasses.dataclass
class ChoiceParam(typing.Generic[CT], Param):
    choices: typing.List[CT]
    max_num_picks: int = 1


@dataclasses.dataclass
class BooleanParameter(Param):
    pass
