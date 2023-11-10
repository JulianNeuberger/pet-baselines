from random import random

import typing

from augment import base, params
from nltk.corpus import wordnet
from data import model
import numpy as np


# Author: Benedikt
class Trafo101Step(base.AugmentationStep):
    name = "101"

    def __init__(self, prob: float = 1, type=True, no_dupl=False, **kwargs):
        super().__init__(**kwargs)
        self.prob = prob
        self.type = type
        self.no_dupl = no_dupl

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.FloatParam(name="prob", min_value=0.0, max_value=1.0),
            params.BooleanParameter(name="type"),
            params.BooleanParameter(name="no_dupl"),
        ]

    def do_augment(self, doc: model.Document) -> model.Document:
        doc = doc.copy()
        changed_words = []
        if self.type:
            pos_nltk = "s"
            pos_tags = ["JJ", "JJS", "JJR", "RB", "RBR", "RBS"]
        else:
            pos_nltk = "n"
            pos_tags = ["NN", "NNS", "NNP", "NNPS"]
        for sentence in doc.sentences:
            for token in sentence.tokens:
                exists = False
                if token.pos_tag in pos_tags and random() < self.prob:
                    if self.no_dupl is True and (token.text in changed_words):
                        exists = True
                    if not exists:
                        syns = wordnet.synsets(token.text, pos_nltk)
                        syns = [syn.name().split(".")[0] for syn in syns]
                        for i in range(len(syns)):
                            if token.text != syns[i]:
                                changed_words.append(token.text)
                                token.text = syns[i]

                                break
        return doc
