import typing
import copy
from data import model
from augment import base, params
from random import random, randint

from transformations import tokenmanager


class TrafoInsertStep(base.AugmentationStep):
    def __init__(self, count_insertions = 1, **kwargs):
        super().__init__(**kwargs)
        self.count_insertions = count_insertions
    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.IntegerParam(name="count_insertions", max_value=20, min_value=1),
        ]

    def do_augment(self, doc2: model.Document) -> model.Document:
        doc = copy.deepcopy(doc2)
        for sentence in doc.sentences:
            counter = 0
            while counter < self.count_insertions:
                ran = randint(0, len(sentence.tokens) - 1)
                text = "Test"
                if sentence.tokens[ran].bio_tag == "O":
                    bio = sentence.tokens[ran].bio_tag
                else:
                    bio = "I-" + tokenmanager.get_bio_tag_short(sentence.tokens[ran].bio_tag)
                tok = model.Token(text, sentence.tokens[ran].index_in_document + 1,
                                  tokenmanager.get_pos_tag([text]),
                                  bio,
                                  sentence.tokens[ran].sentence_index)

                mentions = tokenmanager.get_mentions(doc, ran, sentence.tokens[ran].sentence_index)
                if mentions != []:
                    tokenmanager.create_token(doc, tok, ran + 1, mentions[0])
                else:
                    tokenmanager.create_token(doc, tok, ran + 1)
                counter += 1
        return doc