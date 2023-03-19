import typing

import data
from coref import util


class NaiveCoRefSolver:
    def __init__(self, resolved_tags: typing.List[str], min_mention_overlap: float = .33, ner_strategy: str = 'skip'):
        self._tags = resolved_tags
        self._mention_overlap_threshold = min_mention_overlap
        self._ner_strategy = ner_strategy

    def resolve_co_references(self, documents: typing.List[data.Document]) -> typing.List[data.Document]:
        assert all([len(d.entities) == 0 for d in documents])
        for document in documents:
            resolved_mention_indices = []

            for mention_index, mention in enumerate(document.mentions):
                if mention_index in resolved_mention_indices:
                    continue

                cluster: typing.List[int] = [mention_index]

                for other_index, other in enumerate(document.mentions):
                    if mention == other:
                        continue

                    if other_index in resolved_mention_indices:
                        continue

                    if other.ner_tag not in self._tags:
                        continue

                    mention_token_texts = self._text_from_mention(mention, document)
                    other_token_texts = self._text_from_mention(other, document)

                    overlap_text = self._longest_overlap_of_lists(mention_token_texts, other_token_texts)

                    left_overlap = len(overlap_text) / len(mention_token_texts)
                    right_overlap = len(overlap_text) / len(other_token_texts)
                    overlap = (left_overlap + right_overlap) / 2

                    if overlap < self._mention_overlap_threshold:
                        continue

                    cluster.append(other_index)

                entity = util.resolve_ner_conflicts(document, cluster, self._ner_strategy)
                if entity is None:
                    continue

                resolved_mention_indices.extend(entity.mention_indices)

                document.entities.append(entity)

            util.resolve_remaining_mentions_to_entities(document)

        return documents

    @staticmethod
    def _text_from_mention(mention: data.Mention, document: data.Document) -> typing.List[str]:
        return [t.text for t in mention.get_tokens(document)]

    @staticmethod
    def _longest_overlap_of_lists(left: typing.List, right: typing.List) -> typing.List:
        """
        Adapted from https://www.w3resource.com/python-exercises/list-advanced/python-list-advanced-exercise-11.php
        """
        m, n = len(left), len(right)
        jh = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if left[i - 1] == right[j - 1]:
                    jh[i][j] = 1 + jh[i - 1][j - 1]
                else:
                    jh[i][j] = max(jh[i - 1][j], jh[i][j - 1])

        result = []
        i, j = m, n
        while i > 0 and j > 0:
            if left[i - 1] == right[j - 1]:
                result.append(left[i - 1])
                i -= 1
                j -= 1
            elif jh[i - 1][j] > jh[i][j - 1]:
                i -= 1
            else:
                j -= 1

        return result[::-1]
