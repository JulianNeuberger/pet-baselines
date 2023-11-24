import random
import typing

from transformers import pipeline

from augment import base, params
from data import model
from pos_enum import Pos


class Trafo26Step(base.AugmentationStep):
    def __init__(
        self,
        dataset: typing.List[model.Document],
        n=1,
        tag_groups: typing.List[Pos] = None,
    ):
        super().__init__(dataset)
        self.n = n
        self.unmasker = pipeline("fill-mask", model="xlm-roberta-base", top_k=1)
        self.pos_tags_to_consider: typing.List[str] = [
            v for group in tag_groups for v in group.tags
        ]

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.ChoiceParam(name="tag_groups", choices=list(Pos), max_num_picks=4),
            params.IntegerParam(name="n", min_value=1, max_value=20),
        ]

    @staticmethod
    def mask_sentence(
        doc: model.Document, sentence: model.Sentence, token_to_mask: model.Token
    ) -> str:
        tokens = [t.text for t in sentence.tokens]
        tokens[token_to_mask.index_in_sentence(doc)] = "<mask>"
        return " ".join(tokens)

    def do_augment(self, doc: model.Document) -> model.Document:
        doc = doc.copy()

        candidates: typing.List[model.Token] = []
        for token in doc.tokens:
            if token.pos_tag in self.pos_tags_to_consider:
                candidates.append(token)
        random.shuffle(candidates)

        for candidate in candidates:
            sentence = doc.sentences[candidate.sentence_index]
            masked_sentence = self.mask_sentence(doc, sentence, candidate)
            new_sentence = self.unmasker(masked_sentence)[0]
            doc.tokens[candidate.index_in_document].text = new_sentence["token_str"]

        return doc
