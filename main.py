import functools
import typing

import data
import pipeline
from eval import metrics

FoldStats = typing.List[typing.Dict[str, metrics.Scores]]


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
        p=pipeline.Pipeline(name='complete pipeline rule-based relations', steps=[
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
        p=pipeline.Pipeline(name='complete pipeline cat-boost relations', steps=[
            pipeline.CrfMentionEstimatorStep(name='crf mention extraction'),
            pipeline.NeuralCoReferenceResolutionStep(name='neural coreference resolution',
                                                     resolved_tags=['Actor', 'Activity Data'],
                                                     cluster_overlap=.5,
                                                     mention_overlap=.8,
                                                     ner_strategy='frequency'),
            pipeline.CatBoostRelationExtractionStep(name='cat-boost relation extraction')
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


if __name__ == '__main__':
    main()
