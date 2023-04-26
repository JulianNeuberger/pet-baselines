import abc

from data import model


class AugmentationStep(abc.ABC):
    def do_augment(self, doc: model.Document) -> model.Document:
        raise NotImplementedError()
