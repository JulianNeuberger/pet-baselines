from random import random

import typing

from augment import base, params
from nltk.corpus import wordnet
from data import model
import numpy as np
from pos_enum import Pos


# Author: Benedikt
class Trafo101Step(base.AugmentationStep):
    name = "101"

    def __init__(
        self,
        dataset: typing.List[model.Document],
        prob: float = 1,
        tag_groups: typing.List[Pos] = None,
        no_dupl=False,
    ):
        super().__init__(dataset)
        self.prob = prob
        self.pos_tags_to_consider: typing.List[str] = [
            v for group in tag_groups for v in group.tags
        ]
        self.no_dupl = no_dupl

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.FloatParam(name="prob", min_value=0.0, max_value=1.0),
            params.ChoiceParam(name="tag_groups", choices=list(Pos), max_num_picks=4),
            params.BooleanParameter(name="no_dupl"),
        ]

    def do_augment(self, doc: model.Document) -> model.Document:
        doc = doc.copy()
        changed_words = []
        for sentence in doc.sentences:
            for token in sentence.tokens:
                exists = False
                if token.pos_tag in self.pos_tags_to_consider and random() < self.prob:
                    if self.no_dupl is True and (token.text in changed_words):
                        exists = True
                    if not exists:
                        syns = wordnet.synsets(token.text)
                        syns = [syn.name().split(".")[0] for syn in syns]
                        for i in range(len(syns)):
                            if token.text != syns[i]:
                                changed_words.append(token.text)
                                token.text = syns[i]

                                break
        return doc
