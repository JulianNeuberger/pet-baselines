import dataclasses
import functools
import pathlib
import typing

import coref
import data
import entities
from eval import metrics
import relations

FoldStats = typing.List[typing.Dict[str, metrics.Scores]]


@dataclasses.dataclass
class PipelineConfig:
    crf_model_path: pathlib.Path
    mention_overlap: float
    cluster_overlap: float
    ner_strategy: str


@dataclasses.dataclass
class F1Stats:
    mentions_f1_stats: FoldStats = dataclasses.field(default_factory=list)

    entities_perfect_mentions_f1_stats: FoldStats = dataclasses.field(default_factory=list)
    entities_predicted_mentions_f1_stats: FoldStats = dataclasses.field(default_factory=list)
    entities_perfect_mentions_naive_f1_stats: FoldStats = dataclasses.field(default_factory=list)

    relations_perfect_entities_f1_stats: FoldStats = dataclasses.field(default_factory=list)
    relations_predicted_entities_perfect_mentions_f1_stats: FoldStats = dataclasses.field(default_factory=list)
    relations_predicted_entities_predicted_mentions_f1_stats: FoldStats = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class PipelineResult:
    ground_truth: typing.List[data.Document]

    mentions_baseline: typing.List[data.Document]

    entities_perfect_mentions: typing.List[data.Document]
    entities_perfect_mentions_naive: typing.List[data.Document]
    entities_predicted_mentions: typing.List[data.Document]

    relations_perfect_entities: typing.List[data.Document]
    relations_predicted_entities_perfect_mentions: typing.List[data.Document]
    relations_predicted_entities_predicted_mentions: typing.List[data.Document]


def cross_validation(folds: typing.List[typing.Tuple[typing.List[data.Document], typing.List[data.Document]]],
                     pipeline_config: PipelineConfig) -> F1Stats:
    stats = F1Stats()

    for n_fold, (train_fold, test_fold) in enumerate(folds):
        pipeline_config.crf_model_path = pathlib.Path(f'models/crf.{n_fold}.model')

        result = pipeline(train_data=train_fold, test_data=test_fold,
                          config=pipeline_config)
        stats.mentions_f1_stats.append(metrics.mentions_f1_stats(
            predicted_documents=result.mentions_baseline,
            ground_truth_documents=result.ground_truth
        ))

        stats.entities_perfect_mentions_f1_stats.append(metrics.entity_f1_stats(
            predicted_documents=result.entities_perfect_mentions,
            ground_truth_documents=result.ground_truth,
            min_num_mentions=2
        ))
        stats.entities_predicted_mentions_f1_stats.append(metrics.entity_f1_stats(
            predicted_documents=result.entities_predicted_mentions,
            ground_truth_documents=result.ground_truth,
            min_num_mentions=2
        ))
        stats.entities_perfect_mentions_naive_f1_stats.append(metrics.entity_f1_stats(
            predicted_documents=result.entities_perfect_mentions_naive,
            ground_truth_documents=result.ground_truth,
            min_num_mentions=2
        ))

        stats.relations_perfect_entities_f1_stats.append(metrics.relation_f1_stats(
            predicted_documents=result.relations_perfect_entities,
            ground_truth_documents=result.ground_truth
        ))
        stats.relations_predicted_entities_perfect_mentions_f1_stats.append(metrics.relation_f1_stats(
            predicted_documents=result.relations_predicted_entities_perfect_mentions,
            ground_truth_documents=result.ground_truth
        ))
        stats.relations_predicted_entities_predicted_mentions_f1_stats.append(metrics.relation_f1_stats(
            predicted_documents=result.relations_predicted_entities_predicted_mentions,
            ground_truth_documents=result.ground_truth
        ))

    return stats


def mention_extraction_module(config: PipelineConfig, *,
                              train_data: typing.List[data.Document],
                              test_data: typing.List[data.Document]) -> typing.List[data.Document]:
    # crf step for entity extraction
    print('Running mention extraction...')
    estimator = entities.ConditionalRandomFieldsEstimator(config.crf_model_path)
    estimator.train(train_data)
    mention_extraction_input = [d.copy(clear_mentions=True) for d in test_data]
    return estimator.predict(mention_extraction_input)


def entity_extraction_module(config: PipelineConfig, naive: bool,
                             documents: typing.List[data.Document]) -> typing.List[data.Document]:
    resolved_tags = ['Activity Data', 'Actor']
    if naive:
        solver = coref.NaiveCoRefSolver(resolved_tags,
                                        min_mention_overlap=config.mention_overlap)
    else:
        solver = coref.NeuralCoRefSolver(resolved_tags,
                                         ner_tag_strategy=config.ner_strategy,
                                         min_mention_overlap=config.mention_overlap,
                                         min_cluster_overlap=config.cluster_overlap)
    return solver.resolve_co_references(documents)


def relation_extraction_module(documents: typing.List[data.Document]) -> typing.List[data.Document]:
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
        relations.rules.UsesRelationRule(activity_data_tag=activity_data, activity_tag=activity, uses_relation_tag=uses)
    ])
    return extractor.predict(documents)


def pipeline(config: PipelineConfig, *,
             train_data: typing.List[data.Document], test_data: typing.List[data.Document]):
    mention_extraction_input = [d.copy(clear_mentions=True) for d in test_data]
    baseline_mentions = mention_extraction_module(config, train_data=train_data, test_data=mention_extraction_input)

    entity_extraction_perfect_mentions = [d.copy(clear_entities=True) for d in test_data]
    entities_perfect_mentions = entity_extraction_module(config,
                                                         documents=entity_extraction_perfect_mentions,
                                                         naive=False)

    entity_extraction_predicted_mentions = [d.copy(clear_entities=True) for d in baseline_mentions]
    entities_predicted_mentions = entity_extraction_module(config,
                                                           documents=entity_extraction_predicted_mentions,
                                                           naive=False)

    entity_extraction_perfect_mentions_naive = [d.copy(clear_entities=True) for d in test_data]
    entities_perfect_mentions_naive = entity_extraction_module(config,
                                                               documents=entity_extraction_perfect_mentions_naive,
                                                               naive=True)

    relations_perfect_inputs = [d.copy(clear_relations=True) for d in test_data]
    relations_from_perfect_entities = relation_extraction_module(relations_perfect_inputs)

    relations_predicted_entities_perfect_mentions = [d.copy(clear_relations=True) for d in entities_perfect_mentions]
    relations_from_predicted_entities_perfect_mentions = relation_extraction_module(
        relations_predicted_entities_perfect_mentions)

    relations_predicted_entities_predicted_mentions = [d.copy(clear_relations=True) for d in
                                                       entities_predicted_mentions]
    relations_from_predicted_entities_predicted_mentions = relation_extraction_module(
        relations_predicted_entities_predicted_mentions)

    return PipelineResult(
        ground_truth=test_data,
        mentions_baseline=baseline_mentions,

        entities_perfect_mentions=entities_perfect_mentions,
        entities_predicted_mentions=entities_predicted_mentions,
        entities_perfect_mentions_naive=entities_perfect_mentions_naive,

        relations_perfect_entities=relations_from_perfect_entities,
        relations_predicted_entities_perfect_mentions=relations_from_predicted_entities_perfect_mentions,
        relations_predicted_entities_predicted_mentions=relations_from_predicted_entities_predicted_mentions
    )


def main():
    train_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/test.json') for i in range(5)]

    folds = list(zip(train_folds, test_folds))
    pipeline_config = PipelineConfig(mention_overlap=.8, cluster_overlap=.5,
                                     ner_strategy='frequency', crf_model_path=pathlib.Path())
    stats = cross_validation(folds, pipeline_config=pipeline_config)
    print_f1_stats(stats, order=['activity', 'activity data', 'actor', 'further specification',
                                 'xor gateway', 'and gateway', 'condition specification',
                                 'flow', 'uses', 'actor performer', 'actor recipient',
                                 'further specification', 'same gateway'])


def print_f1_stats(stats: F1Stats, order: typing.List[str]) -> None:
    def _accumulate(left: typing.Dict[str, metrics.Scores],
                    right: typing.Dict[str, metrics.Scores]) -> typing.Dict[str, metrics.Scores]:
        key_set = set(left.keys()).union(set(right.keys()))
        return {
            ner_tag: left.get(ner_tag, metrics.Scores(1, 1, 1)) + right.get(ner_tag, metrics.Scores(1, 1, 1))
            for ner_tag in key_set
        }

    def _print_module_stats(f1_stats: FoldStats, only_for_tags: typing.List[str] = None) -> None:
        num_folds = len(f1_stats)
        accumulated_stats = functools.reduce(_accumulate, f1_stats)
        accumulated_stats = {
            k: v / num_folds for k, v in accumulated_stats.items()
        }

        if only_for_tags:
            only_for_tags = [t.lower() for t in only_for_tags]
            accumulated_stats = {
                k: v for k, v in accumulated_stats.items() if k.lower() in only_for_tags
            }

        len_ner_tags = max([len(t) for t in accumulated_stats.keys()])

        print(f'{" " * (len_ner_tags - 2)}Tag |   P     |   R     |   F1    ')
        print(f'{"=" * (len_ner_tags + 2)}+=========+=========+=============')
        for ner_tag in order:
            if ner_tag not in accumulated_stats:
                continue
            score = accumulated_stats[ner_tag]
            print(f' {ner_tag: >{len_ner_tags}} | {score.p: >7.2%} | {score.r: >7.2%} | {score.f1: >7.2%}')
        print(f'{"-" * (len_ner_tags + 2)}+---------+---------+---------')

        score = sum(accumulated_stats.values(), metrics.Scores(0, 0, 0)) / len(accumulated_stats)
        print(f' {"Overall": >{len_ner_tags}} | {score.p: >7.2%} | {score.r: >7.2%} | {score.f1: >7.2%}')
        print(f'{"-" * (len_ner_tags + 2)}+---------+---------+---------')

    print(f'--- MENTIONS -----------------------------------------------------------')
    _print_module_stats(stats.mentions_f1_stats)
    print()
    print()

    print(f'--- ENTITIES (PERFECT MENTIONS) ----------------------------------------')
    resolved_entity_tags = ['Actor', 'Activity Data']
    _print_module_stats(stats.entities_perfect_mentions_f1_stats, only_for_tags=resolved_entity_tags)
    print()
    print(f'--- ENTITIES (PREDICTED MENTIONS) --------------------------------------')
    _print_module_stats(stats.entities_predicted_mentions_f1_stats, only_for_tags=resolved_entity_tags)
    print()
    print(f'--- ENTITIES NAIVE SOLVER (PERFECT MENTIONS) ---------------------------')
    _print_module_stats(stats.entities_perfect_mentions_naive_f1_stats, only_for_tags=resolved_entity_tags)
    print()
    print()

    print(f'--- RELATIONS (PERFECT ENTITIES) ---------------------------------------')
    _print_module_stats(stats.relations_perfect_entities_f1_stats)
    print()
    print(f'--- RELATIONS (PREDICTED ENTITIES, PERFECT MENTIONS) -------------------')
    _print_module_stats(stats.relations_predicted_entities_perfect_mentions_f1_stats)
    print()
    print(f'--- RELATIONS (PREDICTED ENTITIES, PREDICTED MENTIONS) -----------------')
    _print_module_stats(stats.relations_predicted_entities_predicted_mentions_f1_stats)
    print()
    print()


if __name__ == '__main__':
    main()
