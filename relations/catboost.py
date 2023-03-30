import itertools
import math
import random
import typing

import catboost
import numpy as np

import data


class CatBoostRelationEstimator:
    def __init__(self,
                 name: str,
                 negative_sampling_rate: float,
                 num_trees: int,
                 context_size: int,
                 relation_tags: typing.List[str],
                 ner_tags: typing.List[str],
                 use_pos_features: bool,
                 verbose: bool,
                 seed: int = 42):
        self._model = catboost.CatBoostClassifier(iterations=num_trees, verbose=False, random_state=seed)
        self._negative_rate = negative_sampling_rate
        self._target_tags = [t.lower() for t in relation_tags]
        self._ner_tags = [t.lower() for t in ner_tags]
        self._no_relation_tag = 'NO REL'
        self._verbose = verbose
        self._name = name
        self._context_size = context_size
        self._use_pos_features = use_pos_features
        self._seed = seed

    def train(self, documents: typing.List[data.Document]) -> 'CatBoostRelationEstimator':
        random.seed(self._seed)
        samples = []
        for document in documents:
            samples_in_document = 0
            for relation in document.relations:
                head_entity = document.entities[relation.head_entity_index]
                tail_entity = document.entities[relation.tail_entity_index]
                for head_mention_index, tail_mention_index in itertools.product(head_entity.mention_indices,
                                                                                tail_entity.mention_indices):
                    head_mention = document.mentions[head_mention_index]
                    tail_mention = document.mentions[tail_mention_index]

                    # make sure we can learn from this mention pair,
                    # i.e. they are evidence for the relation between their entities
                    if head_mention.sentence_index not in relation.evidence:
                        continue
                    if tail_mention.sentence_index not in relation.evidence:
                        continue

                    x = self._build_features(head_mention_index, tail_mention_index, document)
                    # y = self._relation_tag_to_one_hot(relation.tag)
                    y = relation.tag
                    samples_in_document += 1
                    samples.append((x, y))
            assert len(document.relations) <= samples_in_document

            for head, tail in self._negative_sample(document, num_positive=samples_in_document):
                x = self._build_features(head, tail, document)
                y = self._no_relation_tag
                # y = self._relation_tag_to_one_hot(self._no_relation_tag)
                samples.append((x, y))

        random.shuffle(samples)

        xs = [x for x, _ in samples]
        ys = [y for _, y in samples]

        num_cat_features = 2
        if self._use_pos_features:
            num_cat_features += 2
        num_cat_features += self._context_size * 4

        cat_features = list(range(2, 2 + num_cat_features))
        try:
            self._model.fit(xs, ys, cat_features=cat_features)
        except catboost.CatboostError as e:
            print('cat features', cat_features)
            print(xs[0])
            raise e

        graph = self._model.plot_tree(self._model.tree_count_ - 1)
        graph.render(filename=f'graphs/{self._name}')

        return self

    def predict(self, documents: typing.List[data.Document]) -> typing.List[data.Document]:
        assert all([len(d.entities) > 0 for d in documents])
        assert all([len(d.relations) == 0 for d in documents])

        for document in documents:
            xs = []
            argument_indices = []
            for head_index, tail_index in list(itertools.combinations(range(len(document.mentions)), 2)):
                x = self._build_features(head_index, tail_index, document)
                xs.append(x)
                argument_indices.append((head_index, tail_index))
                x = self._build_features(tail_index, head_index, document)
                xs.append(x)
                argument_indices.append((tail_index, head_index))

            ys = self._model.predict(xs)
            document.relations = self._get_relations_from_predictions(argument_indices, ys, document)

        return documents

    def _negative_sample(self, document: data.Document, num_positive: int):
        num_negative_samples = math.ceil(self._negative_rate * num_positive)
        negative_samples = []

        candidates = list(itertools.combinations(range(len(document.mentions)), 2))
        candidates += [(t, h) for h, t in candidates]

        for head_mention_index, tail_mention_index in candidates:
            if len(negative_samples) >= num_negative_samples:
                break

            head_index = document.entity_index_for_mention(document.mentions[head_mention_index])
            tail_index = document.entity_index_for_mention(document.mentions[tail_mention_index])
            if document.relation_exists_between(head_index, tail_index):
                continue

            negative_samples.append((head_mention_index, tail_mention_index))

        if len(negative_samples) < num_negative_samples:
            if self._verbose:
                print(f'Could only build {len(negative_samples)}/{num_negative_samples} '
                      f'negative samples, as there were not enough candidates in {document.name}, '
                      f'reusing some.')
            missing_num_samples = num_negative_samples - len(negative_samples)
            while missing_num_samples > 0:
                negative_samples += negative_samples[:missing_num_samples]
                missing_num_samples = num_negative_samples - len(negative_samples)

            random.shuffle(negative_samples)

        return negative_samples

    def _get_relations_from_predictions(self, indices: typing.List[typing.Tuple[int, int]],
                                        ys: np.ndarray, document: data.Document) -> typing.List[data.Relation]:
        assert len(indices) == len(ys)

        relations = []

        for (head_mention_index, tail_mention_index), tag in zip(indices, ys):
            tag = tag[0]
            assert type(tag) == str, f'Expected prediction to be string, got "{tag}" ({type(tag)})'
            if tag == self._no_relation_tag:
                # predicted no relation between the two
                continue

            assert tag in self._target_tags

            head_mention = document.mentions[head_mention_index]
            tail_mention = document.mentions[tail_mention_index]

            head_index = document.entity_index_for_mention(document.mentions[head_mention_index])
            tail_index = document.entity_index_for_mention(document.mentions[tail_mention_index])

            relations.append(data.Relation(
                head_entity_index=head_index,
                tail_entity_index=tail_index,
                tag=tag,
                evidence=list({head_mention.sentence_index, tail_mention.sentence_index})
            ))

        return relations

    def _build_features(self, head_mention_index: int, tail_mention_index: int, document: data.Document) -> typing.List:
        head = document.mentions[head_mention_index]
        tail = document.mentions[tail_mention_index]

        head_pos = head.get_tokens(document)[0].pos_tag[:2]
        tail_pos = tail.get_tokens(document)[0].pos_tag[:2]

        context = []
        for i in range(-self._context_size, self._context_size + 1):
            if i == 0:
                continue

            mention_index = head_mention_index + i
            if mention_index < 0:
                context.append(None)
            elif mention_index >= len(document.mentions):
                context.append(None)
            else:
                context.append(document.mentions[mention_index])

            mention_index = tail_mention_index + i
            if mention_index < 0:
                context.append(None)
            elif mention_index >= len(document.mentions):
                context.append(None)
            else:
                context.append(document.mentions[mention_index])

        features = [
            tail_mention_index - head_mention_index,
            tail.sentence_index - head.sentence_index,
            head.ner_tag,
            tail.ner_tag,
        ]

        if self._use_pos_features:
            features += [
                head_pos,
                tail_pos,
            ]

        features += [
            m.ner_tag if m is not None else '' for m in context
        ]

        return features

    def _ner_tag_to_one_hot(self, ner_tag: str) -> np.ndarray:
        one_hot = np.zeros(len(self._ner_tags), )
        one_hot[self._ner_tags.index(ner_tag.lower())] = 1.0
        return one_hot

    def _relation_tag_to_one_hot(self, relation_tag: str) -> np.ndarray:
        target = np.zeros((len(self._target_tags) + 1,))
        if relation_tag.lower() in self._target_tags:
            target[self._target_tags.index(relation_tag)] = 1.
        else:
            target[len(self._target_tags)] = 1.
        return target

    def _relation_tag_to_scalar(self, relation_tag: str) -> float:
        if relation_tag == self._no_relation_tag:
            return len(self._target_tags)
        return self._target_tags.index(relation_tag.lower())
