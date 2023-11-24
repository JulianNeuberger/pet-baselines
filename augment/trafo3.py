import random
import typing

from nltk.corpus import wordnet

from augment import base, params
from data import model
from transformations import tokenmanager


class Trafo3Step(base.AugmentationStep):
    def __init__(self, dataset: typing.List[model.Document], n: int = 1):
        super().__init__(dataset)
        self.n = n

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.IntegerParam(name="n", min_value=1, max_value=10),
        ]

    @staticmethod
    def get_candidates(document: model.Document) -> typing.List[model.Token]:
        return [t for t in document.tokens if t.pos_tag in ["JJ", "JJR", "JJS"]]

    def do_augment(self, doc: model.Document) -> model.Document:
        doc = doc.copy()
        candidates = self.get_candidates(doc)
        random.shuffle(candidates)

        num_changed = 0
        for candidate in candidates:
            changed = self.antonym_switch(candidate, doc)
            if changed:
                num_changed += 1
            if num_changed == self.n:
                break

        return doc

    @staticmethod
    def antonym_switch(token: model.Token, doc: model.Document) -> bool:
        syn_sets = wordnet.synsets(token.text, "a")
        syn_sets = [s for s in syn_sets if ".a." in s.name()]

        if len(syn_sets) == 0:
            return False

        first_syn_set = syn_sets[0]
        lemma = first_syn_set.lemmas()[0]
        antonyms = lemma.antonyms()

        if len(antonyms) == 0:
            return False

        antonyms.sort(key=lambda x: str(x).split(".")[2])
        antonym = antonyms[0].name()

        antonym_tokens = antonym.split("_")

        tokenmanager.expand_token(doc, token, antonym_tokens)
        return True
