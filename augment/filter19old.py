import typing

import data
from augment import base
import operator

from data import model
from transformations import tokenmanager


# Author: Benedikt
# OLD
class Filter19StepOld(base.AugmentationStep):
    def __init__(
        self, dataset: typing.List[model.Document], triples: typing.List = [()]
    ):
        super().__init__(dataset)
        self.triples = triples

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
    def parse_pos_tags(self, pos_tag):
        pos_tags = {
            "N": ["NN", "NNS", "NNP", "NNPS"],  # Nomen
            "A": ["JJ", "JJR", "JJS", "RB", "RBR", "RBS"],  # Adjektive und Adverben
            "V": ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],  # Verben
        }
        return pos_tags[pos_tag]

    def do_augment(self, doc: model.Document) -> model.Document:
        i = 0
        while i < len(doc.sentences):
            sentence = doc.sentences[i]
            nomen_counter = 0
            ad_counter = 0
            verb_counter = 0
            for token in sentence.tokens:
                if token.pos_tag in Filter19Step.parse_pos_tags(self, "N"):
                    nomen_counter += 1
                if token.pos_tag in Filter19Step.parse_pos_tags(self, "A"):
                    ad_counter += 1
                else:
                    verb_counter += 1
            for triple in self.triples:
                curr_threshold = triple[0]
                curr_op = Filter19Step.parse_operator(triple[1])
                curr_pos_tag = triple[2]
                variable_count = 0
                if curr_pos_tag == "N":
                    variable_count = nomen_counter
                elif curr_pos_tag == "A":
                    variable_count = ad_counter
                else:
                    variable_count = verb_counter
                if curr_op(variable_count, curr_threshold):
                    tokenmanager.delete_sentence(doc, i)
                    i -= 1
                    break
            i += 1
        return doc
