import copy
import typing
from random import shuffle

from augment import base
from data import model


def run_augmentation_old(dataset: typing.List[model.Document], step: base.AugmentationStep) -> typing.List[model.Document]:
    return [step.do_augment(doc) for doc in dataset]


def run_augmentation(dataset: typing.List[model.Document], step: base.AugmentationStep, aug_rate) -> typing.List[model.Document]:
    num_of_doc_to_aug = int(len(dataset) * aug_rate)
    extended_dataset = []
    ds = copy.deepcopy(dataset)
    shuffle(ds)
    for i in range(num_of_doc_to_aug):
        if i < len(ds):
            extended_dataset.append(ds[i])
        else:
            x = i % len(ds)
            extended_dataset.append(ds[x])
    aug_dataset = [step.do_augment(doc) for doc in extended_dataset]
    dataset.extend(aug_dataset)
    shuffle(dataset)
    print(len(dataset))
    return dataset
