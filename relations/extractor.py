import typing

import data
import relations


class RuleBasedRelationEstimator:
    def __init__(self, rules: typing.List[relations.RelationExtractionRule]):
        self._rules = rules

    def predict(self, documents: typing.List[data.Document]) -> typing.List[data.Document]:
        for document in documents:
            assert len(document.relations) == 0

        for document in documents:
            for rule in self._rules:
                rs = rule.get_relations(document)
                for r in rs:
                    assert r.head_entity_index != r.tail_entity_index
                document.relations.extend(rs)

        return documents
