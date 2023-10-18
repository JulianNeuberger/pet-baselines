import data
import augment
import pipeline
import main as console
from transformations import tokenmanager


def augmentation_test():
    train_set = data.loader.read_documents_from_json('./jsonl/fold_0/train.json')
    test_set = data.loader.read_documents_from_json('./jsonl/fold_0/test.json')

    # augmentation_step: augment.AugmentationStep = augment.Trafo82Step(True, True, 1)
    augmentation_step: augment.AugmentationStep = augment.Trafo101Step()
    augmented_train_set = augment.run_augmentation(train_set, augmentation_step)
    augmented_train_set.extend(train_set)
    # relation_extraction_step = pipeline.CatBoostRelationExtractionStep(name='cat-boost relation extraction',
    #                                                                    context_size=2, num_trees=1000,
    #                                                                    negative_sampling_rate=40.0)
    relation_extraction_step = pipeline.RuleBasedRelationExtraction(name='rule-based relation extraction')

    p = pipeline.Pipeline(name='complete-cat-boost', steps=[
        pipeline.CrfMentionEstimatorStep(name='crf mention extraction'),
        pipeline.NeuralCoReferenceResolutionStep(name='neural coreference resolution',
                                                 resolved_tags=['Actor', 'Activity Data'],
                                                 cluster_overlap=.5,
                                                 mention_overlap=.8,
                                                 ner_strategy='frequency'),
        #relation_extraction_step
    ])

    un_augmented_result = p.run(train_documents=train_set, test_documents=test_set, ground_truth_documents=test_set)
    augmented_result = p.run(train_documents=augmented_train_set, test_documents=test_set,
                             ground_truth_documents=test_set)

    print('Un-Augmented:')
    console.print_pipeline_results(p, console.accumulate_pipeline_results([un_augmented_result]))
    print()
    print('-------')
    print()
    print('Augmented:')
    console.print_pipeline_results(p, console.accumulate_pipeline_results([augmented_result]))
    # tokenmanager.save_to_json(p)


def main():
    augmentation_test()


if __name__ == '__main__':
    main()
