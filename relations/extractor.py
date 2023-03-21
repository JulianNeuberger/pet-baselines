import typing

import data
import relations


class RuleBasedRelationEstimator:
    def __init__(self, rules: typing.List[relations.RelationExtractionRule]):
        self._rules = rules

    def predict(self, documents: typing.List[data.Document]) -> typing.List[data.Document]:
        assert all([len(d.entities) > 0 for d in documents])
        assert all([len(d.relations) == 0 for d in documents])

        for document in documents:
            for rule in self._rules:
                rs = rule.get_relations(document)
                for r in rs:
                    assert r.head_entity_index != r.tail_entity_index
                document.relations.extend(rs)

        return documents
