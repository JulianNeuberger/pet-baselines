import spacy
import typing
from transformers import pipeline
from random import random
import data
from data import model
from augment import base, params
import copy
from pos_enum import Pos
class Trafo26Step(base.AugmentationStep):

    def __init__(self, p = 0.5, tag_groups: typing.List[Pos] = None, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.unmasker = pipeline(
            "fill-mask", model="xlm-roberta-base", top_k=1
        )
        self.pos_tags_to_consider: typing.List[str] = [
            v
            for group in tag_groups
            for v in group.tags
        ]

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.ChoiceParam(name="tag_groups", choices=list(Pos), max_num_picks=4),
            params.FloatParam(name="p", min_value=0.0, max_value=1.0),

        ]
    def do_augment(self, doc2: model.Document) -> model.Document:
        doc = copy.deepcopy(doc2)
        for sentence in doc.sentences:
            sentence_string = ""
            changed_id = []
            for i, token in enumerate(sentence.tokens):
                if token.pos_tag in self.pos_tags_to_consider:
                    sentence_string += " <mask>"
                    changed_id.append(i)
                else:
                    sentence_string += f" {token.text}"
            if len(changed_id) == 0:
                continue
            new_sent = self.unmasker(sentence_string)
            new_words = []
            for nw in new_sent:
                if len(changed_id) == 1:
                    new_words.append(nw["token_str"])
                else:
                    new_words.append(nw[0]["token_str"])
            for j, i in enumerate(changed_id):
                if random() >= self.p:
                    sentence.tokens[i].text = new_words[j]
        return doc