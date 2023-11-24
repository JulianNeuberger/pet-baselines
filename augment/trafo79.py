import typing
import copy
from data import model
from augment import base, params
from transformations import tokenmanager
from random import random
from pos_enum import Pos


class Trafo79Step(base.AugmentationStep):
    def __init__(
        self,
        dataset: typing.List[model.Document],
        p=1,
        tag_groups: typing.List[Pos] = None,
    ):
        super().__init__(dataset)
        self.p = p
        self.pos_tags_to_consider: typing.List[str] = [
            v for group in tag_groups for v in group.tags
        ]

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.FloatParam(name="p", min_value=0.0, max_value=1.0),
            params.ChoiceParam(name="tag_groups", choices=list(Pos), max_num_picks=4),
        ]

    def do_augment(self, doc2: model.Document) -> model.Document:
        doc = copy.deepcopy(doc2)
        for sentence in doc.sentences:
            for token in sentence.tokens:
                if random() > self.p and token.pos_tag in self.pos_tags_to_consider:
                    tokenmanager.delete_token(doc, token.index_in_document)
        return doc
