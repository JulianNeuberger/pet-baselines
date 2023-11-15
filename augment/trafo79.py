import typing
import copy
from data import model
from augment import base, params
from transformations import tokenmanager
from random import random

class Trafo79Step(base.AugmentationStep):

    def __init__(self, p = 0.5, pos = 0, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        if pos == 0:
            self.pos_to_change = ["NNS", "NN", "NNP", "NNPS"]
        elif pos == 1:
            self.pos_to_change = ["RB", "RBS", "RBR", "JJ", "JJR", "JJS"]
        else:
            self.pos_to_change = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.FloatParam(name="p", min_value=0.0, max_value=1.0),
            params.IntegerParam(name="pos", min_value=0, max_value=2)
        ]

    def do_augment(self, doc2: model.Document) -> model.Document:
        doc = copy.deepcopy(doc2)
        for sentence in doc.sentences:
            for token in sentence.tokens:
                if random() > self.p and token.pos_tag in self.pos_to_change:
                    tokenmanager.delete_token(doc, token.index_in_document)
        return doc