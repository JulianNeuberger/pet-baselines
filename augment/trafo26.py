import spacy
import typing
from transformers import pipeline
from random import random
import data
from data import model
from augment import base, params
import copy

class Trafo26Step(base.AugmentationStep):

    def __init__(self, p = 0.5, pos=0, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.unmasker = pipeline(
            "fill-mask", model="xlm-roberta-base", top_k=1
        )
        if pos == 0:
            self.pos_to_change = ["NNS", "NN", "NNP", "NNPS"]
        elif pos == 1:
            self.pos_to_change = ["RB", "RBS", "RBR", "JJ", "JJR", "JJS"]
        else:
            self.pos_to_change = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.IntegerParam(name="pos", min_value=0, max_value=2),
            params.FloatParam(name="p", min_value=0.0, max_value=1.0),
        ]
    def do_augment(self, doc2: model.Document) -> model.Document:
        doc = copy.deepcopy(doc2)
        for sentence in doc.sentences:
            sentence_string = ""
            changed_id = []
            for i, token in enumerate(sentence.tokens):
                if token.pos_tag in self.pos_to_change:
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