import copy
import data
import pandas as pd
from experiments import run_experiment, evaluate_experiment_bleu, evaluate_unaugmented_data
import augment
import matplotlib.pyplot as plt
import seaborn as sns

def get_scores_unaugmented():
    train_folds = [data.loader.read_documents_from_json(f'../jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'../jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean"]

    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$'])

    # double the train set
    for j in range(5):
        train_folds[j].extend(train_folds[j])

    # actual training
    f_1_score = run_experiment("Unaugmented Data", train_folds, test_folds)

    # evaluate
    all_scores = evaluate_unaugmented_data(unaug_train_folds=train_folds,
                                          aug_train_folds=train_folds, f_score=f_1_score)

    df_complete = df_complete.append(all_scores[0], ignore_index=True)
    df_complete.to_json(path_or_buf=f"./../experiment_results/unaugmented/{names[0]}.json", indent=4)

    # create jsons
    for k in range(1, len(all_scores)):
        df = all_scores[k]
        df.to_json(path_or_buf=f"./../experiment_results/unaugmented/{names[k]}.json", indent=4)


def experiment1_1():
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'../jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'../jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bleu"]

    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BleuScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(1, 6):
        augmented_train_folds = copy.deepcopy(train_folds)
        augmentation_step: augment.AugmentationStep = augment.Trafo3Step(prob=i/5)  # adapt

        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)

        # actual training
        f_1_score = run_experiment("Experiment 1.1", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bleu(unaug_train_folds=train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./../experiment_results/trafo3/exp3.3/{names[k]}_{i/5}.json", indent=4)

    df_complete.index = ["0.2", "0.4", "0.6", "0.8", "1.0"]
    df_complete.to_json(path_or_buf=f"./../experiment_results/trafo3/exp3.3/{names[0]}.json", indent=4)
    sns.set_theme()
    df_complete.plot(kind="bar")
    plt.show()


experiment1_1()
