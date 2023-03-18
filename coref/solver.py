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

    def __init__(self, co_referencable_tags: typing.List[str]):
        self._tags = [t.lower() for t in co_referencable_tags]

    def resolve_co_references(self, documents: typing.List[data.Document]) -> typing.List[data.Document]:
        for document in documents:
            coref_entities = self._get_co_reference_indices(document)
            for e in coref_entities:
                # try to resolve the list of mentions (each a list of token indices) to a entity
                entity = self._resolve_single_entity(e, document, verbose=True)
                print('=========================')
                if entity is None:
                    continue
                if document.contains_entity(entity):
                    continue
                document.entities.append(entity)
            self._resolve_remaining_mentions_to_entities(document)
        return documents

    @staticmethod
    def _resolve_remaining_mentions_to_entities(document: data.Document) -> None:
        """
        Resolves each mention that is not yet part of an entity
        to a new entity that only contains this mention.
        """
        part_of_entity = set()
        for e in document.entities:
            for i in e.mention_indices:
                part_of_entity.add(i)
        for mention_index, mention in enumerate(document.mentions):
            if mention_index in part_of_entity:
                continue
            document.entities.append(data.Entity([mention_index]))

    def _resolve_single_entity(self,
                               cluster_token_indices: typing.List[typing.List[int]],
                               document: data.Document, verbose: bool = False) -> typing.Optional[data.Entity]:
        if verbose:
            print('neuralcoref found the following cluster:')
            for mention_indices in cluster_token_indices:
                mention_text = " ".join([document.tokens[i].text for i in mention_indices])
                print(f'"{mention_text}", document level token indicies: {mention_indices}')
            print(f'Its tokens are contained in the following predicted mentions (of one of the types {self._tags}:')

        mention_indices = set()
        for mention_token_indices in cluster_token_indices:
            mention_index = self._get_mention_for_token_indices(mention_token_indices, document, self._tags,
                                                                threshold=0.1, verbose=verbose)
            if mention_index is not None:
                mention_indices.add(mention_index)

        mention_indices = list(mention_indices)
        if len(mention_indices) == 0:
            # did not find a single predicted mention for the cluster neuralcoref predicted
            return None
        if len(mention_indices) != len(cluster_token_indices):
            # maybe we should calculate a minimum percentage of resolved cluster entries before we return an
            # entity, this way we could reduce false positives?
            if verbose:
                print(f'Resolved only {len(mention_indices)} of {len(cluster_token_indices)} clusters '
                      f'predicted by neural coref. ' 
                      'This could indicate the predicted cluster was not really of interest to us...')
        else:
            if verbose:
                print(f'Found perfect match of clusters to predicted elements!')
        return data.Entity(mention_indices)

    @staticmethod
    def _get_mention_for_token_indices(token_indices: typing.List[int],
                                       document: data.Document,
                                       valid_tags: typing.List[str],
                                       threshold: float = 0.1,
                                       verbose: bool = False) -> typing.Optional[int]:
        """
        Resolves a list of token indices to a mention predicted by the pipeline

        :param threshold: float between 0 and 1, dictating how many of the given token_indices have
                          to match a mention's token indices for it to be returned

        :returns: index of the matched data.Mention if we could find one, None otherwise
        """
        for mention_index, mention in enumerate(document.mentions):
            if mention.ner_tag.lower() not in valid_tags:
                # this mentions is of a type which we don't do co-references for (e.g. Activity)
                continue

            num_matching_tokens = 0
            for co_ref_mention_index in token_indices:
                co_ref_token = document.tokens[co_ref_mention_index]
                # TODO: currently we mark a cluster as resolved, if one of its token indices are part of the cluster
                # TODO: this could lead to very poor precision, maybe calculate an "overlap" instead?

                if mention.contains_token(co_ref_token, document):
                    # found a predicted mention which is of the correct type and contains a least one
                    # of the cluster's token indices, add it to the list of mentions for the new entity
                    num_matching_tokens += 1

            overlap = num_matching_tokens / len(token_indices)
            is_above_threshold = overlap >= threshold
            if is_above_threshold:
                if verbose:
                    print(f'{mention.pretty_print(document)} with an overlap of {overlap:.2%}')
                return mention_index
        return None

    def _get_co_reference_indices(self, document: data.Document) -> typing.List[typing.List[typing.List[int]]]:
        """
        return a three times nested list
        - 1: list of entities
          - 2: list of mentions of a given entity
            - 3: list of token indices of a given mention
        All token indices are document level!
        """

        clusters: typing.List[neuralcoref.neuralcoref.Cluster]
        clusters = self.nlp([token.text for token in document.tokens])._.coref_clusters
        entities = []
        for cluster in clusters:
            entity = []
            for mention in cluster.mentions:
                entity.append(list(range(mention.start, mention.end)))
            entities.append(entity)
        return entities
