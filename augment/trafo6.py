import copy
from nltk.stem import WordNetLemmatizer
import data
from data import model
from augment import base, params
import spacy
from random import random
import typing
from transformations import tokenmanager


class Trafo6Step(base.AugmentationStep):

    def __init__(self, p = 0.5,  **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.nlp = spacy.load('en_core_web_sm')

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.FloatParam(name="p", min_value=0.0, max_value=1.0),
        ]
    def do_augment(self, doc2: model.Document) -> model.Document:
        doc = copy.deepcopy(doc2)
        for sentence in doc.sentences:
            for i, token in enumerate(sentence.tokens):
                toktok = self.nlp(token.text)
                randNum = random()
                if token.text.lower() == "not" and randNum < self.p:
                    tokenmanager.delete_token(doc, token.index_in_document)
                elif len(toktok) > 1 and randNum < self.p:
                    if toktok[1] == "n't":
                        token.text = toktok[0]
        return doc





