import copy

from augment import base
import operator

from data import model
from transformations import tokenmanager


class Filter10Step(base.AugmentationStep):
    def __init__(self, length: int, op: str, bio: str):
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

    def do_augment(self, doc: model.Document) -> model.Document:
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

filter = Filter10Step(2, "<=")
print(filter.do_augment(doc))