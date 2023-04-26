import typing

from augment import base
from data import model


def run_augmentation(dataset: typing.List[model.Document], step: base.AugmentationStep) -> typing.List[model.Document]:
    return [step.do_augment(doc) for doc in dataset]
