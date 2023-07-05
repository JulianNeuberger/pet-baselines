from augment import base
import operator

from data import model
from transformations import tokenmanager
import copy

class Filter19Step(base.AugmentationStep):
    def __init__(self, length: int=4, op: str=">", pos: str="V"):
        self.length = length
        self.op = Filter19Step.parse_operator(op)
        self.pos = Filter19Step.parse_pos_tags(pos)

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

    @staticmethod
    def parse_pos_tags(pos_tag):
        pos_tags = {
            "N": ["NN", "NNS", "NNP", "NNPS"],  # Nomen
            "A": ["JJ", "JJR", "JJS", "RB", "RBR", "RBS"],  # Adjektive und Adverben
            "V": ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],  # Verben
        }
        return pos_tags[pos_tag]
    def do_augment(self, doc2: model.Document) -> model.Document:
        doc = copy.deepcopy(doc2)
        i = 0
        while i < len(doc.sentences):
            sentence = doc.sentences[i]
            pos_in_sent = 0
            for token in sentence.tokens:
                if token.pos_tag in self.pos:
                    pos_in_sent += 1

            condition = self.op(pos_in_sent, self.length)
            if condition:
                tokenmanager.delete_sentence(doc, i)
                i -= 1
            i += 1
        return doc