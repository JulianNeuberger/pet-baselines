import abc
import dataclasses
import pathlib
import typing

import coref
import data
import mentions
import relations
from eval import metrics


@dataclasses.dataclass
class PipelineStepResult:
    predictions: typing.List[data.Document]
    stats: typing.Dict[str, metrics.Stats]


class PipelineStep(abc.ABC):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self):
        return self._name

    def __hash__(self):
        return hash(self._name)

    def __eq__(self, other):
        if not type(other) == PipelineStep:
            return False
        return self._name == other._name

    def run(self, *,
            train_documents: typing.List[data.Document],
            test_documents: typing.List[data.Document],
            ground_truth_documents: typing.List[data.Document]):
        train_data = [d.copy() for d in train_documents]
        test_data = [d.copy() for d in test_documents]
        result = self._run(train_documents=train_data, test_documents=test_data)
        stats = self._eval(ground_truth=ground_truth_documents, predictions=result)
        return PipelineStepResult(result, stats)

    def _eval(self, *,
              predictions: typing.List[data.Document],
              ground_truth: typing.List[data.Document]) -> typing.Dict[str, metrics.Stats]:
        raise NotImplementedError()

    def _run(self, *,
             train_documents: typing.List[data.Document],
             test_documents: typing.List[data.Document]) -> typing.List[data.Document]:
        raise NotImplementedError()


class CatBoostRelationExtractionStep(PipelineStep):
    def __init__(self, name: str, num_trees: int, negative_sampling_rate: float, context_size: int,
                 use_pos_features: bool = False,
                 verbose: bool = False, seed: int = 42):
        super().__init__(name)
        self._num_trees = num_trees
        self._negative_sampling = negative_sampling_rate
        self._context_size = context_size
        self._verbose = verbose
        self._seed = seed
        self._use_pos_features = use_pos_features

    def _eval(self, *, predictions: typing.List[data.Document],
              ground_truth: typing.List[data.Document]) -> typing.Dict[str, metrics.Stats]:
        return metrics.relation_f1_stats(predicted_documents=predictions, ground_truth_documents=ground_truth,
                                         verbose=self._verbose)

    def _run(self, *, train_documents: typing.List[data.Document],
             test_documents: typing.List[data.Document]) -> typing.List[data.Document]:
        ner_tags = ['Activity', 'Actor', 'Activity Data', 'Condition Specification',
                    'Further Specification', 'AND Gateway', 'XOR Gateway']
        relation_tags = ['Flow', 'Uses', 'Actor Performer', 'Actor Recipient', 'Further Specification', 'Same Gateway']
        estimator = relations.CatBoostRelationEstimator(negative_sampling_rate=self._negative_sampling,
                                                        num_trees=self._num_trees,
                                                        use_pos_features=self._use_pos_features,
                                                        context_size=self._context_size,
                                                        relation_tags=relation_tags,
                                                        ner_tags=ner_tags,
                                                        name=self._name,
                                                        seed=self._seed,
                                                        verbose=False)
        estimator.train(train_documents)
        test_documents = [d.copy(clear_relations=True) for d in test_documents]
        return estimator.predict(test_documents)


class RuleBasedRelationExtraction(PipelineStep):
    def _eval(self, *, predictions: typing.List[data.Document],
              ground_truth: typing.List[data.Document]) -> typing.Dict[str, metrics.Stats]:
        return metrics.relation_f1_stats(predicted_documents=predictions, ground_truth_documents=ground_truth)

    def _run(self, *, train_documents: typing.List[data.Document],
             test_documents: typing.List[data.Document]) -> typing.List[data.Document]:
        activity = 'Activity'
        actor = 'Actor'
        activity_data = 'Activity Data'
        condition = 'Condition Specification'
        further_spec = 'Further Specification'
        and_gateway = 'AND Gateway'
        xor_gateway = 'XOR Gateway'

        flow = 'Flow'
        uses = 'Uses'
        performer = 'Actor Performer'
        recipient = 'Actor Recipient'
        further_spec_relation = 'Further Specification'
        same_gateway = 'Same Gateway'

        extractor = relations.RuleBasedRelationEstimator([
            relations.rules.SameGatewayRule(triggering_elements=[xor_gateway, and_gateway], target_tag=same_gateway),
            relations.rules.GatewayActivityRule(gateway_tags=[and_gateway, xor_gateway], activity_tag=activity,
                                                same_gateway_tag=same_gateway, flow_tag=flow),
            relations.rules.SequenceFlowsRule(triggering_elements=[activity, xor_gateway, and_gateway, condition],
                                              target_tag=flow),
            relations.rules.ActorPerformerRecipientRule(actor_tag=actor, activity_tag=activity,
                                                        performer_tag=performer, recipient_tag=recipient),
            relations.rules.FurtherSpecificationRule(further_specification_element_tag=further_spec,
                                                     further_specification_relation_tag=further_spec_relation,
                                                     activity_tag=activity),
            relations.rules.UsesRelationRule(activity_data_tag=activity_data, activity_tag=activity,
                                             uses_relation_tag=uses)
        ])
        test_documents = [d.copy(clear_relations=True) for d in test_documents]
        return extractor.predict(test_documents)


class CrfMentionEstimatorStep(PipelineStep):
    def _eval(self, *,
              predictions: typing.List[data.Document],
              ground_truth: typing.List[data.Document]) -> typing.Dict[str, metrics.Stats]:
        return metrics.mentions_f1_stats(predicted_documents=predictions, ground_truth_documents=ground_truth)

    def _run(self, *,
             train_documents: typing.List[data.Document],
             test_documents: typing.List[data.Document]) -> typing.List[data.Document]:
        estimator = mentions.ConditionalRandomFieldsEstimator(pathlib.Path(f'models/crf/{self._name}'))
        estimator.train(train_documents)
        mention_extraction_input = [d.copy(clear_mentions=True) for d in test_documents]
        return estimator.predict(mention_extraction_input)


class NeuralCoReferenceResolutionStep(PipelineStep):
    def __init__(self, name: str, resolved_tags: typing.List[str],
                 ner_strategy: str, mention_overlap: float, cluster_overlap: float):
        super().__init__(name)
        self._resolved_tags = resolved_tags
        self._ner_strategy = ner_strategy
        self._mention_overlap = mention_overlap
        self._cluster_overlap = cluster_overlap

    def _eval(self, *,
              predictions: typing.List[data.Document],
              ground_truth: typing.List[data.Document]) -> typing.Dict[str, metrics.Stats]:
        return metrics.entity_f1_stats(predicted_documents=predictions,
                                       only_tags=self._resolved_tags,
                                       ground_truth_documents=ground_truth)

    def _run(self, *,
             train_documents: typing.List[data.Document],
             test_documents: typing.List[data.Document]) -> typing.List[data.Document]:
        test_documents = [d.copy(clear_entities=True) for d in test_documents]
        solver = coref.NeuralCoRefSolver(self._resolved_tags,
                                         ner_tag_strategy=self._ner_strategy,
                                         min_mention_overlap=self._mention_overlap,
                                         min_cluster_overlap=self._cluster_overlap)
        return solver.resolve_co_references(test_documents)


class NaiveCoReferenceResolutionStep(PipelineStep):
    def __init__(self, name: str, resolved_tags: typing.List[str], mention_overlap: float):
        super().__init__(name)
        self._resolved_tags = resolved_tags
        self._mention_overlap = mention_overlap

    def _eval(self, *,
              predictions: typing.List[data.Document],
              ground_truth: typing.List[data.Document]) -> typing.Dict[str, metrics.Stats]:
        return metrics.entity_f1_stats(predicted_documents=predictions,
                                       only_tags=self._resolved_tags,
                                       ground_truth_documents=ground_truth)

    def _run(self, *,
             train_documents: typing.List[data.Document],
             test_documents: typing.List[data.Document]) -> typing.List[data.Document]:
        test_documents = [d.copy(clear_entities=True) for d in test_documents]
        solver = coref.NaiveCoRefSolver(self._resolved_tags, min_mention_overlap=self._mention_overlap)
        return solver.resolve_co_references(test_documents)
