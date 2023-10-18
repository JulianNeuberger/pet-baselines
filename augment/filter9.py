import copy

from augment import base
import operator
from transformations import tokenmanager
from data import model

# Author: Benedikt
class Filter9Step(base.AugmentationStep):
    def __init__(self, length: int=12, op: str="<"):
        self.length = length
        self.op = Filter9Step.parse_operator(op)

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
            condition = self.op(len(sentence.tokens), self.length)
            print(condition)
            if condition:
                tokenmanager.delete_sentence(doc, i)
                i -= 1
            i += 1
        return doc