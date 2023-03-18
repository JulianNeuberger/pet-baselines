import typing

import data
import neuralcoref
import spacy


class CoRefSolver:
    # loading an english SpaCy model
    spacy.cli.download('en_core_web_md')
    nlp = spacy.load('en_core_web_md')

    # load NeuralCoref and add it to the pipe of SpaCy's model
    coref = neuralcoref.NeuralCoref(nlp.vocab)
    nlp.add_pipe(coref, name='neuralcoref')
    nlp.tokenizer = nlp.tokenizer.tokens_from_list

    def __init__(self):
        pass

    def resolve_co_references(self, documents: typing.List[data.Document]) -> typing.List[data.Document]:
        for document in documents:
            coref_entities = self._get_co_reference_indices(document)
            for e in coref_entities:
                entity = self._resolve_single_entity(e, document)
                if entity is None:
                    continue
                if document.contains_entity(entity):
                    continue
                document.entities.append(entity)
        return documents

    @staticmethod
    def _resolve_single_entity(list_of_mention_indices: typing.List[typing.List[int]],
                               document: data.Document) -> typing.Optional[data.Entity]:
        mention_indices = set()
        for co_ref_mention_indices in list_of_mention_indices:
            for co_ref_mention_index in co_ref_mention_indices:
                for mention_index, mention in enumerate(document.mentions):
                    if co_ref_mention_index in mention.token_indices:
                        mention_indices.add(mention_index)
                        continue
        mention_indices = list(mention_indices)
        if len(mention_indices) == 0:
            return None
        return data.Entity(mention_indices)

    def _get_co_reference_indices(self, document: data.Document) -> typing.List[typing.List[typing.List[int]]]:
        # return a three times nested list
        # - first level: list of entities
        #   - second level: list of mentions of a given entity
        #     - third level: list of token indices of a given mention

        clusters: typing.List[neuralcoref.neuralcoref.Cluster]
        clusters = self.nlp([token.text for token in document.tokens])._.coref_clusters
        entities = []
        for cluster in clusters:
            entity = []
            for mention in cluster.mentions:
                entity.append(list(range(mention.start, mention.end)))
            entities.append(entity)
        return entities
