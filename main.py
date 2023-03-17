import dataclasses
import functools
import pathlib
import typing

import data
import entities
import metrics
import relations


@dataclasses.dataclass
class PipelineResult:
    ground_truth: typing.List[data.Document]
    predictions_baseline_1: typing.List[data.Document]
    predictions_baseline_2: typing.List[data.Document]
    predictions_baseline_3: typing.List[data.Document]


def cross_validation(folds: typing.List[typing.Tuple[typing.List[data.Document], typing.List[data.Document]]]):
    baseline_1_f1_stats = []
    baseline_2_f1_stats = []
    baseline_3_f1_stats = []

    for n_fold, (train_fold, test_fold) in enumerate(folds):
        result = pipeline(train_fold, test_fold, crf_model_path=pathlib.Path(f'models/crf.{n_fold}.model'))
        baseline_1_f1_stats.append(metrics.mentions_f1_stats(
            predicted_documents=result.predictions_baseline_1,
            ground_truth_documents=result.ground_truth
        ))
        baseline_2_f1_stats.append(metrics.relation_f1_stats(
            predicted_documents=result.predictions_baseline_2,
            ground_truth_documents=result.ground_truth
        ))
        baseline_3_f1_stats.append(metrics.relation_f1_stats(
            predicted_documents=result.predictions_baseline_3,
            ground_truth_documents=result.ground_truth
        ))

    print(f'Fold |   P     |   R     |   F1    ')
    print(f'=====+=========+=========+=========')
    print(f'--- B1 ----------------------------')
    _print_f1_stats(baseline_1_f1_stats)
    print(f'--- B2 ----------------------------')
    _print_f1_stats(baseline_2_f1_stats)
    print(f'--- B3 ----------------------------')
    _print_f1_stats(baseline_3_f1_stats)


def pipeline(train_data: typing.List[data.Document], test_data: typing.List[data.Document], *,
             crf_model_path: pathlib.Path):
    # BASELINE 1 - ENTITY EXTRACTION VIA CRF ########################################################################
    # crf step for entity extraction
    estimator = entities.ConditionalRandomFieldsEstimator(crf_model_path)
    estimator.train(train_data)
    baseline_1_input = [d.copy() for d in test_data]
    predictions_baseline_1 = estimator.predict(baseline_1_input)

    # relation extraction step
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

    # BASELINE 2 - RELATIONS ON PERFECT ENTITIES ####################################################################
    documents_with_perfect_entities = []
    for d in test_data:
        d = d.copy()
        d.relations = []
        documents_with_perfect_entities.append(d)
    predictions_baseline_2 = extractor.predict(documents_with_perfect_entities)

    # BASELINE 3 - RELATIONS ON BASELINE 1 PREDICTIONS ##############################################################
    baseline_3_input = [d.copy() for d in predictions_baseline_1]
    predictions_baseline_3 = extractor.predict(baseline_3_input)

    # BASELINE 4 - CO-REFERENCES ####################################################################################

    return PipelineResult(
        ground_truth=test_data,
        predictions_baseline_1=predictions_baseline_1,
        predictions_baseline_2=predictions_baseline_2,
        predictions_baseline_3=predictions_baseline_3,
    )


def main():
    train_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/test.json') for i in range(5)]

    folds = list(zip(train_folds, test_folds))
    cross_validation(folds)


def _print_f1_stats(f1_stats: typing.List[typing.Tuple[float, float, float]]) -> None:
    for n_fold, (p, r, f1) in enumerate(f1_stats):
        print(f'   {n_fold + 1} | {p: >7.2%} | {r: >7.2%} | {f1: >7.2%}')
    p, r, f1 = functools.reduce(lambda item, total: (total[0] + item[0], total[1] + item[1], total[2] + item[2]),
                                f1_stats)
    num_folds = len(f1_stats)
    print(f'-----+---------+---------+---------')
    print(f'     | {p / num_folds: >7.2%} | {r / num_folds: >7.2%} | {f1 / num_folds: >7.2%}')


if __name__ == '__main__':
    main()
