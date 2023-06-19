import copy
import typing

from augment import base
import operator

from data import model
from transformations import tokenmanager


class Filter19Step(base.AugmentationStep):
    def __init__(self, triples: typing.List):
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
            "N": ["NN", "NNS", "NNP", "NNPS"], # Nomen
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


tokens = [model.Token(text="I", index_in_document=0,
                      pos_tag="PRP", bio_tag="B-Actor",
                      sentence_index=0),
          model.Token(text="leave", index_in_document=1,
                      pos_tag="VBP", bio_tag="O",
                      sentence_index=0),
          model.Token(text="Human", index_in_document=2,
                      pos_tag="NN",
                      bio_tag="B-Activity Data",
                      sentence_index=0),
          model.Token(text="resources", index_in_document=3,
                      pos_tag="NNS", bio_tag="I-Activity Data",
                      sentence_index=0),
          model.Token(text="easy", index_in_document=4,
                      pos_tag="JJ", bio_tag="O",
                      sentence_index=0),
          model.Token(text=".", index_in_document=5,
                      pos_tag=".", bio_tag=".",
                      sentence_index=0)
          ]

sentence1 = model.Sentence(tokens=tokens)
tokens2 = [model.Token(text="hello", index_in_document=6,
                      pos_tag="PRP", bio_tag="B-Actor",
                      sentence_index=1),
          model.Token(text="my", index_in_document=7,
                      pos_tag="VBP", bio_tag="O",
                      sentence_index=1),
          model.Token(text="Human", index_in_document=8,
                      pos_tag="NN",
                      bio_tag="B-Activity Data",
                      sentence_index=1),
          model.Token(text=".", index_in_document=9,
                      pos_tag=".", bio_tag=".",
                      sentence_index=1)
          ]
sentence2 = model.Sentence(tokens=tokens2)
doc = model.Document(
    text="I leave HR office",
    name="1", sentences=[sentence1, sentence2],
    mentions=[],
    entities=[],
    relations=[])

filter = Filter19Step([(1, "<=", "N")])
print(filter.do_augment(doc))