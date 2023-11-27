import typing

import nltk.tokenize

from augment import base, params
from data import model
from transformations import tokenmanager


class BaseAbbreviationStep(base.AugmentationStep):
    def __init__(
        self,
        dataset: typing.List[model.Document],
        abbreviations: typing.Dict[str, str],
        case_sensitive: bool = False,
    ):
        super().__init__(dataset)
        self.case_sensitive = case_sensitive
        self.expansions = abbreviations
        self.contractions = {v: k for k, v in abbreviations.items()}

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return []

    def get_contraction_candidates(
        self, doc: model.Document
    ) -> typing.List[typing.List[model.Token]]:
        return self._get_candidates(self.contractions, doc)

    def get_expansion_candidates(
        self, doc: model.Document
    ) -> typing.List[typing.List[model.Token]]:
        return self._get_candidates(self.expansions, doc)

    @staticmethod
    def _get_candidates(
        dictionary: typing.Dict[str, str], doc: model.Document
    ) -> typing.List[typing.List[model.Token]]:
        candidates = []
        candidate: typing.List[model.Token] = []
        for token in doc.tokens:
            candidate += [token]
            candidate_key = " ".join(t.text for t in candidate)
            if candidate_key in dictionary:
                candidates.append(candidate)
                candidate = []
                continue
            if BaseAbbreviationStep.has_keys_starting_with(dictionary, candidate_key):
                candidate += [token]
                continue
            candidate = []
        return candidates

    @staticmethod
    def has_keys_starting_with(
        dictionary: typing.Dict[str, typing.Any], partial_key: str
    ) -> bool:
        for key in dictionary.keys():
            if key.startswith(partial_key):
                return True
        return False

    def do_augment(self, doc: model.Document) -> model.Document:
        doc = doc.copy()

        expansion_candidates = self.get_expansion_candidates(doc)
        contraction_candidates = self.get_contraction_candidates(doc)

        self.replace_candidates(doc, expansion_candidates, self.expansions)
        self.replace_candidates(doc, contraction_candidates, self.contractions)

        return doc

    @staticmethod
    def replace_candidates(
        doc: model.Document,
        candidates: typing.List[typing.List[model.Token]],
        lookup_table: typing.Dict[str, str],
    ):
        for candidate in candidates:
            start = candidate[0].index_in_sentence(doc)
            stop = candidate[-1].index_in_sentence(doc) + 1
            key = " ".join(t.text for t in candidate)
            replace_text = lookup_table[key]
            replace_tokens = nltk.tokenize.word_tokenize(replace_text)
            tokenmanager.replace_sequence_text_in_sentence(
                doc, candidate[0].sentence_index, start, stop, replace_tokens
            )

    def load_bank110(self):
        sep = "\t"
        temp_acronyms = []
        contracted = []
        expanded = []
        with open(
            "./transformations/trafo82/acronyms.tsv", "r", encoding="utf-8"
        ) as file:
            for line in file:
                key, value = line.strip().split(sep)
                # temp_acronyms[key] = value
                contracted.append(key)
                expanded.append(value)
        # Place long keys first to prevent overlapping
        acronyms = {}
        for k in sorted(temp_acronyms, key=len, reverse=True):
            acronyms[k] = temp_acronyms[k]
        acronyms = acronyms
        return contracted, expanded
