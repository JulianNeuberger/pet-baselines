import dataclasses
import functools
import pathlib
import typing

import coref
import data
import entities
from eval import metrics
import relations


@dataclasses.dataclass
class PipelineResult:
    ground_truth: typing.List[data.Document]

    mentions_baseline: typing.List[data.Document]

    entities_perfect_mentions: typing.List[data.Document]
    entities_predicted_mentions: typing.List[data.Document]

    relations_perfect_entities: typing.List[data.Document]
    relations_predicted_entities_perfect_mentions: typing.List[data.Document]
    relations_predicted_entities_predicted_mentions: typing.List[data.Document]


def cross_validation(folds: typing.List[typing.Tuple[typing.List[data.Document], typing.List[data.Document]]]):
    mentions_f1_stats = []

    entities_perfect_mentions_f1_stats = []
    entities_predicted_mentions_f1_stats = []

    relations_perfect_entities_f1_stats = []
    relations_predicted_entities_perfect_mentions_f1_stats = []
    relations_predicted_entities_predicted_mentions_f1_stats = []

    for n_fold, (train_fold, test_fold) in enumerate(folds):
        result = pipeline(train_fold, test_fold, crf_model_path=pathlib.Path(f'models/crf.{n_fold}.model'))
        mentions_f1_stats.append(metrics.mentions_f1_stats(
            predicted_documents=result.mentions_baseline,
            ground_truth_documents=result.ground_truth
        ))

        entities_perfect_mentions_f1_stats.append(metrics.entity_f1_stats(
            predicted_documents=result.entities_perfect_mentions,
            ground_truth_documents=result.ground_truth
        ))
        entities_predicted_mentions_f1_stats.append(metrics.entity_f1_stats(
            predicted_documents=result.entities_predicted_mentions,
            ground_truth_documents=result.ground_truth
        ))

        relations_perfect_entities_f1_stats.append(metrics.relation_f1_stats(
            predicted_documents=result.relations_perfect_entities,
            ground_truth_documents=result.ground_truth
        ))
        relations_predicted_entities_perfect_mentions_f1_stats.append(metrics.relation_f1_stats(
            predicted_documents=result.relations_predicted_entities_perfect_mentions,
            ground_truth_documents=result.ground_truth
        ))
        relations_predicted_entities_predicted_mentions_f1_stats.append(metrics.relation_f1_stats(
            predicted_documents=result.relations_predicted_entities_predicted_mentions,
            ground_truth_documents=result.ground_truth
        ))


    print(f'Fold |   P     |   R     |   F1    ')
    print(f'=====+=========+=========+==============================================')
    print(f'--- MENTIONS -----------------------------------------------------------')
    _print_f1_stats(mentions_f1_stats)
    print()

    print(f'Fold |   P     |   R     |   F1    ')
    print(f'=====+=========+=========+==============================================')
    print(f'--- ENTITIES (PERFECT MENTIONS) ----------------------------------------')
    _print_f1_stats(entities_perfect_mentions_f1_stats)
    print(f'--- ENTITIES (PREDICTED MENTIONS) --------------------------------------')
    _print_f1_stats(entities_predicted_mentions_f1_stats)
    print()

    print(f'Fold |   P     |   R     |   F1    ')
    print(f'=====+=========+=========+==============================================')
    print(f'--- RELATIONS (PERFECT ENTITIES) ---------------------------------------')
    _print_f1_stats(relations_perfect_entities_f1_stats)
    print(f'--- RELATIONS (PREDICTED ENTITIES, PERFECT MENTIONS) -------------------')
    _print_f1_stats(relations_predicted_entities_perfect_mentions_f1_stats)
    print(f'--- RELATIONS (PREDICTED ENTITIES, PREDICTED MENTIONS) -----------------')
    _print_f1_stats(relations_predicted_entities_predicted_mentions_f1_stats)
    print()


def pipeline(train_data: typing.List[data.Document], test_data: typing.List[data.Document], *,
             crf_model_path: pathlib.Path):
    # MENTION EXTRACTION VIA CRF (BELLAN) ########################################################################
    # crf step for entity extraction
    print('Running mention extraction...')
    estimator = entities.ConditionalRandomFieldsEstimator(crf_model_path)
    estimator.train(train_data)
    mention_extraction_input = [d.copy(clear_mentions=True) for d in test_data]
    baseline_mentions = estimator.predict(mention_extraction_input)

    # ENTITY EXTRACTION BASELINES (CO-REF RESOLUTION) ###############################################################
    print('Running co-reference resolution on perfect data...')


    resolved_tags = ['Activity Data', 'Actor']
    solver = coref.NeuralCoRefSolver(resolved_tags,
                                     ner_tag_strategy='frequency',
                                     min_mention_overlap=.8,
                                     min_cluster_overlap=.5)
    # solver = coref.NaiveCoRefSolver(resolved_tags, min_mention_overlap=.1, ner_strategy='frequency')

    entity_extraction_perfect_mentions = [d.copy(clear_entities=True) for d in test_data]
    entities_perfect_mentions = solver.resolve_co_references(entity_extraction_perfect_mentions)

    entity_extraction_predicted_mentions = [d.copy(clear_entities=True) for d in baseline_mentions]
    entities_predicted_mentions = solver.resolve_co_references(entity_extraction_predicted_mentions)

    # RELATION EXTRACTION BASELINES #################################################################################
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

    relations_perfect_inputs = [d.copy(clear_relations=True) for d in test_data]
    relations_from_perfect_entities = extractor.predict(relations_perfect_inputs)

    relations_predicted_entities_perfect_mentions = [d.copy(clear_relations=True)
                                                     for d in entities_perfect_mentions]
    relations_from_predicted_entities_perfect_mentions = extractor.predict(relations_predicted_entities_perfect_mentions)

    relations_predicted_entities_predicted_mentions = [d.copy(clear_relations=True)
                                                       for d in entities_predicted_mentions]
    relations_from_predicted_entities_predicted_mentions = extractor.predict(
        relations_predicted_entities_predicted_mentions
    )

    return PipelineResult(
        ground_truth=test_data,
        mentions_baseline=baseline_mentions,

        entities_perfect_mentions=entities_perfect_mentions,
        entities_predicted_mentions=entities_predicted_mentions,

        relations_perfect_entities=relations_from_perfect_entities,
        relations_predicted_entities_perfect_mentions=relations_from_predicted_entities_perfect_mentions,
        relations_predicted_entities_predicted_mentions=relations_from_predicted_entities_predicted_mentions
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
