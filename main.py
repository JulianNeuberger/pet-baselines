import dataclasses
import functools
import pathlib
import typing

import data
import pipeline
from eval import metrics

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
    relations_perfect_entities_base_line_f1_stats: FoldStats = dataclasses.field(default_factory=list)
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
    relations_perfect_entities_base_line: typing.List[data.Document]
    relations_predicted_entities_perfect_mentions: typing.List[data.Document]
    relations_predicted_entities_predicted_mentions: typing.List[data.Document]


def cross_validation(folds: typing.List[typing.Tuple[typing.List[data.Document], typing.List[data.Document]]],
                     pipeline_config: PipelineConfig) -> F1Stats:
    stats = F1Stats()

    for n_fold, (train_fold, test_fold) in enumerate(folds):
        pipeline_config.crf_model_path = pathlib.Path(f'models/crf/{n_fold}.model')

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
        stats.relations_perfect_entities_base_line_f1_stats.append(metrics.relation_f1_stats(
            predicted_documents=result.relations_perfect_entities_base_line,
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


def modular_pipeline(steps: typing.List[pipeline.PipelineStep], *,
                     train_documents: typing.List[data.Document],
                     test_documents: typing.List[data.Document]) -> typing.List[typing.Tuple[typing.List[data.Document],
                                                                                             typing.Dict[
                                                                                                 str, metrics.Scores]]]:
    ret = []
    for step in steps:
        step_train_data = [d.copy() for d in train_documents]
        step_test_data = [d.copy() for d in test_documents]
        prediction, f1_metrics = step.run(train_documents=step_train_data, test_documents=step_test_data)
        ret.append((prediction, f1_metrics))
    return ret


# def _pipeline(config: PipelineConfig, *,
#              train_data: typing.List[data.Document], test_data: typing.List[data.Document]):
#     mention_extraction_input = [d.copy(clear_mentions=True) for d in test_data]
#     baseline_mentions = mention_extraction_module(config, train_data=train_data, test_data=mention_extraction_input)
#
#     entity_extraction_perfect_mentions = [d.copy(clear_entities=True) for d in test_data]
#     entities_perfect_mentions = entity_extraction_module(config,
#                                                          documents=entity_extraction_perfect_mentions,
#                                                          naive=False)
#
#     entity_extraction_predicted_mentions = [d.copy(clear_entities=True) for d in baseline_mentions]
#     entities_predicted_mentions = entity_extraction_module(config,
#                                                            documents=entity_extraction_predicted_mentions,
#                                                            naive=False)
#
#     entity_extraction_perfect_mentions_naive = [d.copy(clear_entities=True) for d in test_data]
#     entities_perfect_mentions_naive = entity_extraction_module(config,
#                                                                documents=entity_extraction_perfect_mentions_naive,
#                                                                naive=True)
#
#     relations_perfect_inputs = [d.copy(clear_relations=True) for d in test_data]
#     relations_from_perfect_entities = relation_extraction_module(rule_based=False, train_documents=train_data,
#                                                                  test_documents=relations_perfect_inputs)
#
#     relations_perfect_inputs = [d.copy(clear_relations=True) for d in test_data]
#     relations_perfect_entities_base_line = relation_extraction_module(rule_based=True, train_documents=train_data,
#                                                                       test_documents=relations_perfect_inputs)
#
#     relations_predicted_entities_perfect_mentions = [d.copy(clear_relations=True) for d in entities_perfect_mentions]
#     relations_from_predicted_entities_perfect_mentions = relation_extraction_module(rule_based=False,
#                                                                                     train_documents=train_data,
#                                                                                     test_documents=relations_predicted_entities_perfect_mentions)
#
#     relations_predicted_entities_predicted_mentions = [d.copy(clear_relations=True) for d in
#                                                        entities_predicted_mentions]
#     relations_from_predicted_entities_predicted_mentions = relation_extraction_module(rule_based=False,
#                                                                                       train_documents=train_data,
#                                                                                       test_documents=relations_predicted_entities_predicted_mentions)
#
#     return PipelineResult(
#         ground_truth=test_data,
#         mentions_baseline=baseline_mentions,
#
#         entities_perfect_mentions=entities_perfect_mentions,
#         entities_predicted_mentions=entities_predicted_mentions,
#         entities_perfect_mentions_naive=entities_perfect_mentions_naive,
#
#         relations_perfect_entities=relations_from_perfect_entities,
#         relations_perfect_entities_base_line=relations_perfect_entities_base_line,
#         relations_predicted_entities_perfect_mentions=relations_from_predicted_entities_perfect_mentions,
#         relations_predicted_entities_predicted_mentions=relations_from_predicted_entities_predicted_mentions
#     )


def cross_validate_pipeline(p: pipeline.Pipeline, *,
                            train_folds: typing.List[typing.List[data.Document]],
                            test_folds: typing.List[typing.List[data.Document]]):
    assert len(train_folds) == len(test_folds)
    pipeline_results = []
    for n_fold, (train_fold, test_fold) in enumerate(zip(train_folds, test_folds)):
        ground_truth = [d.copy() for d in test_fold]
        pipeline_result = p.run(train_documents=train_fold,
                                test_documents=test_fold,
                                ground_truth_documents=ground_truth)
        pipeline_results.append(pipeline_result)
    res = accumulate_pipeline_results(pipeline_results)
    print_pipeline_results(p, res)
    return res


def accumulate_pipeline_results(pipeline_results: typing.List[pipeline.PipelineResult]) \
        -> typing.Dict[pipeline.PipelineStep, typing.Dict[str, metrics.Scores]]:
    res_by_step: typing.Dict[pipeline.PipelineStep, typing.List[typing.Dict[str, metrics.Scores]]] = {}
    for r in pipeline_results:
        for step, scores in r.step_results.items():
            if step not in res_by_step:
                res_by_step[step] = []
            res_by_step[step].append(scores.f1_metrics)

    num_runs = len(pipeline_results)
    return {
        step: {
            k: v / num_runs for k, v in functools.reduce(_accumulate, f1_stats).items()
        }
        for step, f1_stats
        in res_by_step.items()
    }


def print_pipeline_results(p: pipeline.Pipeline,
                           res: typing.Dict[pipeline.PipelineStep, typing.Dict[str, metrics.Scores]]):
    print(f'=== {p.name} {"=" * (47 - len(p.name))}')
    for step, scores_by_ner in res.items():
        print(f'--- {step.name} {"-" * (47 - len(step.name))}')
        _print_scores(scores_by_ner)


def main():
    train_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/test.json') for i in range(5)]

    cross_validate_pipeline(
        p=pipeline.Pipeline(name='', steps=[
            pipeline.CrfMentionEstimatorStep(name='crf mention extraction'),
            pipeline.NeuralCoReferenceResolutionStep(name='neural coreference resolution',
                                                     resolved_tags=['Actor', 'Activity Data'],
                                                     cluster_overlap=.5,
                                                     mention_overlap=.8,
                                                     ner_strategy='frequency'),
            pipeline.RuleBasedRelationExtraction(name='rule-based relation extraction')
        ]),
        train_folds=train_folds,
        test_folds=test_folds
    )

    print()
    print('---')
    print()

    cross_validate_pipeline(
        p=pipeline.Pipeline(name='relations isolated (rule-based)', steps=[
            pipeline.RuleBasedRelationExtraction(name='perfect entities')
        ]),
        train_folds=train_folds,
        test_folds=test_folds
    )

    print()
    print('---')
    print()

    cross_validate_pipeline(
        p=pipeline.Pipeline(name='relations isolated (cat-boost)', steps=[
            pipeline.CatBoostRelationExtractionStep(name='perfect entities')
        ]),
        train_folds=train_folds,
        test_folds=test_folds
    )

    #
    #
    #
    # folds = list(zip(train_folds, test_folds))
    # pipeline_config = PipelineConfig(mention_overlap=.8, cluster_overlap=.5,
    #                                  ner_strategy='frequency', crf_model_path=pathlib.Path())
    #
    # order = ['activity', 'activity data', 'actor', 'further specification',
    #          'xor gateway', 'and gateway', 'condition specification',
    #          'flow', 'uses', 'actor performer', 'actor recipient',
    #          'further specification', 'same gateway']
    #
    #
    #
    # steps = [
    #     lambda train, test: relation_extraction_module(train_documents=train, test_documents=test, rule_based=False)
    # ]
    # results: FoldStats = []
    # for n_fold, (train_fold, test_fold) in enumerate(folds):
    #     test_fold = [d.copy() for d in test_fold]
    #     step_output, step_stats = modular_pipeline(steps,
    #                                                lambda t, p: metrics.relation_f1_stats(predicted_documents=p,
    #                                                                                       ground_truth_documents=t,
    #                                                                                       verbose=False),
    #                                                train_documents=train_fold, test_documents=test_fold)[0]
    #     results.append(step_stats)
    # print_module_f1_stats(results, 'relations (catboost)', order)
    #
    # steps = [
    #     lambda train, test: relation_extraction_module(train_documents=train, test_documents=test, rule_based=True)
    # ]
    # results: FoldStats = []
    # for n_fold, (train_fold, test_fold) in enumerate(folds):
    #     test_fold = [d.copy() for d in test_fold]
    #     step_output, step_stats = modular_pipeline(steps,
    #                                                lambda t, p: metrics.relation_f1_stats(predicted_documents=p,
    #                                                                                       ground_truth_documents=t,
    #                                                                                       verbose=False),
    #                                                train_documents=train_fold, test_documents=test_fold)[0]
    #     results.append(step_stats)
    # print_module_f1_stats(results, 'relations (rule-based)', order)

    # stats = cross_validation(folds, pipeline_config=pipeline_config)
    # print_f1_stats(stats, )


def _print_scores(scores_by_ner: typing.Dict[str, metrics.Scores], order: typing.List[str] = None):
    len_ner_tags = max([len(t) for t in scores_by_ner.keys()])

    if order is None:
        order = list(scores_by_ner.keys())

    print(f'{" " * (len_ner_tags - 2)}Tag |   P     |   R     |   F1    ')
    print(f'{"=" * (len_ner_tags + 2)}+=========+=========+========')

    for ner_tag in order:
        if ner_tag not in scores_by_ner:
            continue
        score = scores_by_ner[ner_tag]
        print(f' {ner_tag: >{len_ner_tags}} | {score.p: >7.2%} | {score.r: >7.2%} | {score.f1: >7.2%}')
    print(f'{"-" * (len_ner_tags + 2)}+---------+---------+---------')

    score = sum(scores_by_ner.values(), metrics.Scores(0, 0, 0)) / len(scores_by_ner)
    print(f' {"Overall": >{len_ner_tags}} | {score.p: >7.2%} | {score.r: >7.2%} | {score.f1: >7.2%}')
    print(f'{"-" * (len_ner_tags + 2)}+---------+---------+---------')


def _print_module_stats(f1_stats: FoldStats, order: typing.List[str], only_for_tags: typing.List[str] = None) -> None:
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

    _print_scores(accumulated_stats)


def _accumulate(left: typing.Dict[str, metrics.Scores],
                right: typing.Dict[str, metrics.Scores]) -> typing.Dict[str, metrics.Scores]:
    key_set = set(left.keys()).union(set(right.keys()))
    return {
        ner_tag: left.get(ner_tag, metrics.Scores(1, 1, 1)) + right.get(ner_tag, metrics.Scores(1, 1, 1))
        for ner_tag in key_set
    }


def print_module_f1_stats(stats: FoldStats, name: str,
                          order: typing.List[str], only_for_tags: typing.List[str] = None):
    print(f'--- {name} -------------------------------------------------------------')
    _print_module_stats(stats, order, only_for_tags=only_for_tags)
    print()
    print()


def print_f1_stats(stats: F1Stats, order: typing.List[str]) -> None:
    print(f'--- MENTIONS -----------------------------------------------------------')
    _print_module_stats(stats.mentions_f1_stats, order)
    print()
    print()

    print(f'--- ENTITIES (PERFECT MENTIONS) ----------------------------------------')
    resolved_entity_tags = ['Actor', 'Activity Data']
    _print_module_stats(stats.entities_perfect_mentions_f1_stats, order, only_for_tags=resolved_entity_tags)
    print()
    print(f'--- ENTITIES (PREDICTED MENTIONS) --------------------------------------')
    _print_module_stats(stats.entities_predicted_mentions_f1_stats, order, only_for_tags=resolved_entity_tags)
    print()
    print(f'--- ENTITIES NAIVE SOLVER (PERFECT MENTIONS) ---------------------------')
    _print_module_stats(stats.entities_perfect_mentions_naive_f1_stats, order, only_for_tags=resolved_entity_tags)
    print()
    print()

    print(f'--- RELATIONS FOREST (PERFECT ENTITIES) --------------------------------')
    _print_module_stats(stats.relations_perfect_entities_f1_stats, order)
    print()
    print(f'--- RELATIONS RULE-BASED (PERFECT ENTITIES) ----------------------------')
    _print_module_stats(stats.relations_perfect_entities_base_line_f1_stats, order)
    print()
    print(f'--- RELATIONS (PREDICTED ENTITIES, PERFECT MENTIONS) -------------------')
    _print_module_stats(stats.relations_predicted_entities_perfect_mentions_f1_stats, order)
    print()
    print(f'--- RELATIONS (PREDICTED ENTITIES, PREDICTED MENTIONS) -----------------')
    _print_module_stats(stats.relations_predicted_entities_predicted_mentions_f1_stats, order)
    print()
    print()


if __name__ == '__main__':
    main()
