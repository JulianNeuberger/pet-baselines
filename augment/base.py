import abc
import inspect

from augment import params
import typing

from data import model


class AugmentationStep(abc.ABC):
    def __init__(self, dataset: typing.List[model.Document], **kwargs):
        self.dataset = dataset

    def do_augment(self, doc: model.Document) -> model.Document:
        raise NotImplementedError()

    @staticmethod
    def validate_params(clazz: typing.Type["AugmentationStep"]):
        args = inspect.getfullargspec(clazz.__init__).args
        missing_args = []
        for param in clazz.get_params():
            found_arg = False
            for arg in args:
                if param.name == arg:
                    found_arg = True
                    break
            if not found_arg:
                missing_args.append(param)
        if len(missing_args) > 0:
            raise TypeError(f"Missing arguments in __init__ method of {clazz.__name__}: {missing_args}")

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        raise NotImplementedError()
