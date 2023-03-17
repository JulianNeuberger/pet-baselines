import typing

import data


class CoRefSolver:
    def __init__(self):
        pass

    def resolve_co_references(self, documents: typing.List[data.Document]) -> typing.List[data.Document]:
        for document in documents:
            co_reference_token_indices = self._get_co_reference_indices(document)
            for co_reference_tokens in co_reference_token_indices:
                co_referencing_mentions = [(i, document.get_mentions_by_token_index(i)) for i in co_reference_tokens]
                # remove None values
                co_referencing_mentions = [(i, m) for (i, m) in co_referencing_mentions if m is not None]
                if len(co_referencing_mentions) == 0:
                    # found co-reference that contains no tagged elements, i.e., process elements
                    continue

                # TODO: how do we handle entities, where mentions have heterogeneous tags?
                ner_tag = co_referencing_mentions[0][1].ner_tag
                assert all([m.ner_tag == ner_tag for i, m in co_referencing_mentions])

                document.entities.append(data.Entity(
                    ner_tag=ner_tag,
                    mention_indices=[i for i, m in co_referencing_mentions]
                ))
        return documents

    def _get_co_reference_indices(self, document: data.Document) -> typing.List[typing.List[int]]:
        return []