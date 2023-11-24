import random
import typing

from augment import base, params
from data import model
from transformations import tokenmanager


class Trafo24Step(base.AugmentationStep):
    def __init__(self, dataset: typing.List[model.Document], n: int):
        super().__init__(dataset)
        self.n = n

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.IntegerParam(name="n", min_value=1, max_value=20),
        ]

    def do_augment(self, doc: model.Document) -> model.Document:
        for _ in range(self.n):
            num_sentences = len(doc.sentences)
            if num_sentences == 1:
                return doc

            first_index = random.randint(0, num_sentences - 1)

            self.merge_sentences(first_index, doc)

    @staticmethod
    def merge_sentences(first_index: int, doc: model.Document):
        second_index = first_index + 1

        first_sentence = doc.sentences[first_index]
        second_sentence = doc.sentences[first_index + 1]

        new_sentence = first_sentence.copy()
        tokenmanager.delete_token(
            doc, index_in_document=new_sentence.tokens[-1].index_in_document
        )
        new_sentence.tokens += second_sentence.tokens

        first_sentence_length = len(new_sentence.tokens)

        for mention in doc.mentions:
            if mention.sentence_index == first_index + 1:
                mention.sentence_index = first_index
                mention.token_indices = [
                    t + first_sentence_length for t in mention.token_indices
                ]

        for relation in doc.relations:
            if second_index in relation.evidence:
                relation.evidence = [
                    first_index if e is second_index else e for e in relation.evidence
                ]

        doc.sentences[first_index] = new_sentence
        doc.sentences.pop(second_index)
