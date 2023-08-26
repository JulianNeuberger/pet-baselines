import copy

import nltk
import numpy as np

import data
import pandas as pd
from experiments_trafo.experiments import run_experiment, evaluate_experiment_bleu, evaluate_unaugmented_data, \
    evaluate_experiment_bert, evaluate_experiment_with_rate, evaluate_experiment_with_rate_bleu, \
    evaluate_experiment_test, evaluate_experiment_bert_filter, run_experiment_re, run_experiment_crf
import augment

def run_exp101():
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bleu", "ttr_un", "ucer_un", "ttr_mean_un",
             "ucer_mean_un"]
    doubled_train_folds = []
    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(
        columns=['F1 CRF', 'F1 Neural', 'F1 Relation', 'TTR', 'UCER', 'BleuScore'])
    df_entities = pd.DataFrame(
        columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                 'Condition Specification', 'AND Gateway'])
    # augment the dataset - for i in range of the parameter
    rate = np.linspace(0,10,101)
    #rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    for i in rate:
        augmented_train_folds = copy.deepcopy(train_folds)
        unaugmented_train_folds = copy.deepcopy(train_folds)
        augmentation_step: augment.AugmentationStep = augment.Trafo101Step(prob=0.5)  # adapt

        # actual augmentation
        for j in range(5):
            augmented_train_sets = augment.run_augmentation(augmented_train_folds[j], augmentation_step, aug_rate=i)
            augmented_train_set = augmented_train_sets[0]
            unaugmented_train_set = augmented_train_sets[1]
            # augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)
            unaugmented_train_folds[j] = copy.deepcopy(unaugmented_train_set)
            # train_fold = copy.deepcopy(train_folds[j])
            # train_fold.extend(train_folds[j])
            # doubled_train_folds.append(train_fold)

        # actual training
        f_1_scores = run_experiment("Experiment 101.1", augmented_train_folds, test_folds)
        df_entities = df_entities.append(f_1_scores[3], ignore_index=True)
        # evaluation
        all_scores = evaluate_experiment_with_rate_bleu(unaug_train_folds=unaugmented_train_folds,
                                                   aug_train_folds=augmented_train_folds, f_score_crf=f_1_scores[0],
                                                   f_score_neural=f_1_scores[1],
                                                   f_score_rel=f_1_scores[2])

        df_complete = df_complete.append(all_scores[0], ignore_index=True)
        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./paper_results/trafo101_rate/splitted_results/{names[k]}_{i}.json", indent=4)

    # df_complete.index = ["0.0", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0"]
    ix = []
    for i in rate:
        ix.append(str(round(i, 2)))
    df_complete.index = ix
    #df_complete.index = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "1.25", "1.5",
    #                     "1.75", "2.0", "3.0", "4.0", "5.0", "7.0", "10.0"]
    # df_complete.index = ["0.0", "1.0", "2.0", "3.0",  "4.0"]
    # df_entities.index = ["0.0", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0"]
    df_entities.index = ix
    #df_entities.index = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "1.25", "1.5",
     #                    "1.75", "2.0", "3.0", "4.0", "5.0", "7.0", "10.0"]
    # df_entities.index = ["0.0", "1.0", "2.0", "3.0",  "4.0"]
    df_entities.to_json(path_or_buf="./paper_results/trafo101_rate/all_entities_f1.json", indent=4)
    df_complete.to_json(path_or_buf=f"./paper_results/trafo101_rate/{names[0]}.json", indent=4)


run_exp101()