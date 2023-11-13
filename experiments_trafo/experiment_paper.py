import copy
import typing

import nltk
import numpy as np

import data
import pandas as pd
from experiments_trafo.experiments import run_experiment, evaluate_experiment_bleu, evaluate_unaugmented_data, \
    evaluate_experiment_bert, evaluate_experiment_with_rate, evaluate_experiment_with_rate_bleu, \
    evaluate_experiment_test, evaluate_experiment_bert_filter, run_experiment_re, run_experiment_crf
import augment



def run_exp(aug_step_list, rate):
    results = []
    for augmentation_step in aug_step_list:

        # Get the data for augmenting and training
        train_folds = [data.loader.read_documents_from_json(f'../jsonl/fold_{i}/train.json') for i in range(5)]
        test_folds = [data.loader.read_documents_from_json(f'../jsonl/fold_{i}/test.json') for i in range(5)]
        names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bleu", "ttr_un", "ucer_un", "ttr_mean_un",
                 "ucer_mean_un"]
        # specific for this experiment
        df_complete: pd.DataFrame = pd.DataFrame(
            columns=['F1 CRF', 'F1 Neural', 'F1 Relation', 'TTR', 'UCER', 'BleuScore'])
        df_entities = pd.DataFrame(
            columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                     'Condition Specification', 'AND Gateway'])
        # augment the dataset - for i in range of the parameter

        for i in rate:
            augmented_train_folds = copy.deepcopy(train_folds)
            unaugmented_train_folds = copy.deepcopy(train_folds)

            # actual augmentation
            for j in range(5):
                augmented_train_sets = augment.run_augmentation(augmented_train_folds[j], augmentation_step, aug_rate=i)
                augmented_train_set = augmented_train_sets[0]
                unaugmented_train_set = augmented_train_sets[1]
                augmented_train_folds[j] = copy.deepcopy(augmented_train_set)
                unaugmented_train_folds[j] = copy.deepcopy(unaugmented_train_set)

            # actual training
            f_1_scores = run_experiment(f"Transformation Augmentationsrate {i}", augmented_train_folds, test_folds)
            df_entities = df_entities.append(f_1_scores[3], ignore_index=True)
            # evaluation
            all_scores = evaluate_experiment_with_rate_bleu(unaug_train_folds=unaugmented_train_folds,
                                                       aug_train_folds=augmented_train_folds, f_score_crf=f_1_scores[0],
                                                       f_score_neural=f_1_scores[1],
                                                       f_score_rel=f_1_scores[2])

            df_complete = df_complete.append(all_scores[0], ignore_index=True)
            for k in range(1, len(all_scores) - 1):
                df = all_scores[k]

                # hier würden die einzelnen Daten, welche für die Means verwendet werden abgespeichert werden
                # df.to_json(path_or_buf=f"./paper_results/trafo101_rate/splitted_results/{names[k]}_{i}.json", indent=4)

        ix = []
        for i in rate: # runden der Augrate, damit Dataframe Labels passen
            ix.append(str(round(i, 2)))
        df_complete.index = ix

        df_entities.index = ix
        results.append((df_complete, df_entities, augmentation_step.name))
    return results

def control_experiments(step):
    rate = np.linspace(1, 1, 1)
    aug_step_list = []

    aug_step_list.append(step)
    results = run_exp(aug_step_list, rate)
    for result in results:

        result[0].to_json(path_or_buf=f"./paper_results/all_means/all_means{result[2]}.json", indent=4)
        result[1].to_json(path_or_buf=f"./paper_results/all_entities/all_entities_f1{result[2]}.json", indent=4)


step: augment.AugmentationStep = augment.Trafo101Step(prob=0.5)
step2: augment.AugmentationStep = augment.Trafo8Step()
complete = data.loader.read_documents_from_json("complete.json")
#control_experiments(step2)
not_counter = 0
for doc in complete:
    for sentence in doc.sentences:
        for tok in sentence.tokens:
            if tok.text.lower() ==  "not":
                not_counter += 1