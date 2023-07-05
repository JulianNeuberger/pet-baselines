from augment import base
import operator

from data import model
from transformations import tokenmanager
import copy

class Filter10Step(base.AugmentationStep):
    def __init__(self, length: int=4, op: str=">", bio: str="Activity"):
        self.length = length
        self.op = Filter10Step.parse_operator(op)
        self.bio = bio

    @staticmethod
    def parse_operator(op):
        ops = {
            ">": operator.gt,
            "<": operator.lt,
            ">=": operator.ge,
            "<=": operator.le,
            "==": operator.eq,
        }
        return ops[op]

    def do_augment(self, doc2: model.Document) -> model.Document:
        doc = copy.deepcopy(doc2)
        i = 0
        while i < len(doc.sentences):
            sentence = doc.sentences[i]
            named_entities_in_sentence = 0
            for token in sentence.tokens:
                if tokenmanager.get_bio_tag_short(token.bio_tag) == self.bio:
                    named_entities_in_sentence += 1

            condition = self.op(named_entities_in_sentence, self.length)
            if condition:
                tokenmanager.delete_sentence(doc, i)
                i -= 1
            i += 1
        return doc