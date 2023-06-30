import copy
import typing
from random import shuffle

from augment import base
from data import model


def run_augmentation_old(dataset: typing.List[model.Document], step: base.AugmentationStep) -> typing.List[model.Document]:
    return [step.do_augment(doc) for doc in dataset]


def run_augmentation(dataset: typing.List[model.Document], step: base.AugmentationStep, aug_rate):
    num_of_doc_to_aug = int(len(dataset) * aug_rate)
    extended_dataset = []
    ds = copy.deepcopy(dataset)
    indices = []
    for i in range(len(dataset)):
        indices.append(i)
    shuffle(indices)
    #shuffle(ds)
    for i in range(num_of_doc_to_aug):
        if i < len(ds):
            index = indices[i]
            extended_dataset.append(ds[index])
        else:

            x = i % len(ds)
            index = indices[x]
            extended_dataset.append(ds[index])

    unaug_dataset = copy.deepcopy(dataset)
    ext_ds = copy.deepcopy(extended_dataset)
    unaug_dataset.extend(ext_ds)
    print(len(unaug_dataset))
    aug_dataset = [step.do_augment(doc) for doc in extended_dataset]
    aug_data = copy.deepcopy(dataset)
    aug_data.extend(aug_dataset)
    print(len(aug_data))
    indices2 = []
    for i in range(len(unaug_dataset)):
        indices2.append(i)
    print(len(indices2))
    shuffle(indices2)
    unaug_dataset_shuff = copy.deepcopy(unaug_dataset)
    aug_dataset_shuff = copy.deepcopy(aug_data)
    for i in range(len(indices2)):
        unaug_dataset_shuff.append(unaug_dataset[indices2[i]])
        aug_dataset_shuff.append(aug_data[indices2[i]])
    print("------------")
    print(len(unaug_dataset_shuff))
    print(len(aug_dataset_shuff))
    #shuffle(dataset)
    #print(len(dataset))
    return aug_dataset_shuff, unaug_dataset_shuff
