import typing
import copy
from data import model
from augment import base, params
from transformations import tokenmanager
from random import random

class Trafo24Step(base.AugmentationStep):

    def __init__(self, p = 0.5,  **kwargs):
        super().__init__(**kwargs)
        self.p = p

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.FloatParam(name="p", min_value=0.0, max_value=1.0),
        ]

    def do_augment(self, doc2: model.Document) -> model.Document:
        doc = copy.deepcopy(doc2)
        i = 0
        while i < len(doc.sentences):
            if random() > self.p:
                sent_length = len(doc.sentences[i].tokens) - 1
                if i < len(doc.sentences) - 1:
                    # transfer Mentions
                    for mention in doc.mentions:
                        if mention.sentence_index == i + 1:
                            mention.sentence_index -= 1
                            for ment_id in mention.token_indices:
                                ment_id += sent_length
                    # transfer Tokens
                    tok_arr = []
                    for j, token in enumerate(doc.sentences[i + 1].tokens):
                        tok = model.Token(token.text, token.index_in_document - 1, token.pos_tag, token.bio_tag,
                                          token.sentence_index - 1)
                        tok_arr.append(tok)
                    # delete sentence
                    tokenmanager.delete_sentence(doc, i + 1)

                    for j, tok in enumerate(tok_arr):
                        tokenmanager.create_token(doc, tok, sent_length + j)

                    # delete punct
                    tokenmanager.delete_token(doc, tok_arr[-1].index_in_document + 1)
            i += 1
        return doc

