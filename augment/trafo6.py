import copy
import typing
from random import random

import spacy

from augment import base, params
from data import model
from transformations import tokenmanager


class Trafo6Step(base.AugmentationStep):
    def __init__(self, dataset: typing.List[model.Document], p=0.5):
        super().__init__(dataset)
        self.p = p
        self.nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.FloatParam(name="p", min_value=0.0, max_value=1.0),
        ]

    def do_augment(self, doc: model.Document) -> model.Document:
        doc = doc.copy()
        for sentence in doc.sentences:
            for token in sentence.tokens:
                if random() >= self.p:
                    continue

                if token.text.lower() == "not":
                    tokenmanager.delete_token(doc, token.index_in_document)
                    continue

                spacy_document = self.nlp(token.text)
                if len(spacy_document) <= 1:
                    continue

                if spacy_document[1].text == "n't":
                    token.text = spacy_document[0].text
        return doc
