import typing

import data
import neuralcoref
import spacy


class CoRefSolver:
    # loading an english SpaCy model
    nlp = spacy.load('en')

    # load NeuralCoref and add it to the pipe of SpaCy's model
    coref = neuralcoref.NeuralCoref(nlp.vocab)
    nlp.add_pipe(coref, name='neuralcoref')
    nlp.tokenizer = nlp.tokenizer.tokens_from_list

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
                # TODO: LA: Currently they are ignored.
                ner_tag = co_referencing_mentions[0][1].ner_tag
                # assert all([m.ner_tag == ner_tag for i, m in co_referencing_mentions])
                if all([m.ner_tag == ner_tag for i, m in co_referencing_mentions]):
                    document.entities.append(data.Entity(
                        mention_indices=[i for i, m in co_referencing_mentions]
                    ))
        return documents

    def _get_co_reference_indices(self, document: data.Document) -> typing.List[typing.List[int]]:
        coreference_clusters = self.nlp([token.text for token in document.tokens])._.coref_clusters
        coreferences_token_indices = []
        for cluster in coreference_clusters:
            clustered_indices = []
            for mention in cluster.mentions:
                clustered_indices.extend([i for i in range(mention.start, mention.end)])
            coreferences_token_indices.append(clustered_indices)

        return coreferences_token_indices
