import copy
import data
import pandas as pd
from experiments_trafo.experiments import run_experiment, evaluate_experiment_bleu, evaluate_unaugmented_data, \
    evaluate_experiment_bert
import augment
import matplotlib.pyplot as plt
import seaborn as sns


def get_scores_unaugmented():
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
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
    df_complete.to_json(path_or_buf=f"./experiment_results/unaugmented/{names[0]}.json", indent=4)

    # create jsons
    for k in range(1, len(all_scores)):
        df = all_scores[k]
        df.to_json(path_or_buf=f"./experiment_results/unaugmented/{names[k]}.json", indent=4)


def experiment1_1():
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
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
            df.to_json(path_or_buf=f"./experiment_results/trafo3/exp3.3/{names[k]}_{i/5}.json", indent=4)

    df_complete.index = ["0.2", "0.4", "0.6", "0.8", "1.0"]
    df_complete.to_json(path_or_buf=f"./experiment_results/trafo3/exp3.3/{names[0]}.json", indent=4)


def experiment101_1():  # Probability of replacement
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bert"]
    doubled_train_folds = []
    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BertScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(1, 11):
        augmented_train_folds = copy.deepcopy(train_folds)
        augmentation_step: augment.AugmentationStep = augment.Trafo101Step(prob=i/10)  # adapt

        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)
            train_fold = copy.deepcopy(train_folds[j])
            train_fold.extend(train_folds[j])
            doubled_train_folds.append(train_fold)

        # actual training
        f_1_score = run_experiment("Experiment 101.1", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bert(unaug_train_folds=doubled_train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/trafo101/exp101.1/{names[k]}_{i/10}.json", indent=4)

    df_complete.index = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    df_complete.to_json(path_or_buf=f"./experiment_results/trafo101/exp101.1/{names[0]}.json", indent=4)


def experiment101_2():  # Type of replacement
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bert"]
    doubled_train_folds = []
    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BertScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(2):
        if i == 0:
            pos_type = True
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Trafo101Step(type=pos_type)  # adapt
        else:
            pos_type = False
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Trafo101Step(type=pos_type)  # adapt


        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)
            train_fold = copy.deepcopy(train_folds[j])
            train_fold.extend(train_folds[j])
            doubled_train_folds.append(train_fold)

        # actual training
        f_1_score = run_experiment("Experiment 101.2", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bert(unaug_train_folds=doubled_train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/trafo101/exp101.2/{names[k]}_{pos_type}.json", indent=4)

    df_complete.index = ["Adjektive", "Nomen"]
    df_complete.to_json(path_or_buf=f"./experiment_results/trafo101/exp101.2/{names[0]}.json", indent=4)


def experiment101_3():  # if duplicates are allowed
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bert"]
    doubled_train_folds = []
    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BertScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(2):
        if i == 0:
            dupl = True
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Trafo101Step(type=dupl)  # adapt
        else:
            dupl = False
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Trafo101Step(type=dupl)  # adapt


        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)
            train_fold = copy.deepcopy(train_folds[j])
            train_fold.extend(train_folds[j])
            doubled_train_folds.append(train_fold)

        # actual training
        f_1_score = run_experiment("Experiment 101.3", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bert(unaug_train_folds=doubled_train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/trafo101/exp101.3/{names[k]}_{dupl}.json", indent=4)

    df_complete.index = ["True", "False"]
    df_complete.to_json(path_or_buf=f"./experiment_results/trafo3/exp101.3/{names[0]}.json", indent=4)


def experiment33_1():  # Probability of replacement
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bert"]
    doubled_train_folds = []
    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BertScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(1, 11):
        augmented_train_folds = copy.deepcopy(train_folds)
        augmentation_step: augment.AugmentationStep = augment.Trafo33Step(p=i/10)  # adapt

        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)
            train_fold = copy.deepcopy(train_folds[j])
            train_fold.extend(train_folds[j])
            doubled_train_folds.append(train_fold)

        # actual training
        f_1_score = run_experiment("Experiment 33.1", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bert(unaug_train_folds=doubled_train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/trafo33/exp33.1/{names[k]}_{i/10}.json", indent=4)

    df_complete.index = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    df_complete.to_json(path_or_buf=f"./experiment_results/trafo33/exp33.1/{names[0]}.json", indent=4)


def experiment58_1():  # Probability of replacement
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bert"]
    doubled_train_folds = []
    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BertScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(1, 11):
        augmented_train_folds = copy.deepcopy(train_folds)
        augmentation_step: augment.AugmentationStep = augment.Trafo58Step(p=i/10)  # adapt

        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)
            train_fold = copy.deepcopy(train_folds[j])
            train_fold.extend(train_folds[j])
            doubled_train_folds.append(train_fold)

        # actual training
        f_1_score = run_experiment("Experiment 58.1", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bert(unaug_train_folds=doubled_train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/trafo58/exp58.1/{names[k]}_{i/10}.json", indent=4)

    df_complete.index = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    df_complete.to_json(path_or_buf=f"./experiment_results/trafo58/exp58.1/{names[0]}.json", indent=4)


def experiment58_2():  # Language
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bert"]
    doubled_train_folds = []
    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BertScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(5):
        language = ""
        if i == 0:
            language = "de"
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Trafo58Step(lang=language)  # adapt
        elif i == 1:
            language = "es"
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Trafo58Step(lang=language)  # adapt
        elif i == 2:
            language = "zh"
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Trafo58Step(lang=language)  # adapt
        elif i == 3:
            language = "fr"
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Trafo58Step(lang=language)  # adapt
        else:
            language = "ru"
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Trafo58Step(lang=language)  # adapt


        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)
            train_fold = copy.deepcopy(train_folds[j])
            train_fold.extend(train_folds[j])
            doubled_train_folds.append(train_fold)

        # actual training
        f_1_score = run_experiment("Experiment 58.2", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bert(unaug_train_folds=doubled_train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/trafo58/exp58.2/{names[k]}_{language}.json", indent=4)

    df_complete.index = ["Deutsch", "Spanisch", "Chinesisch", "Französisch", "Russisch"]
    df_complete.to_json(path_or_buf=f"./experiment_results/trafo58/exp58.2/{names[0]}.json", indent=4)


def experiment5_1():  # Probability of replacement
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bert"]
    doubled_train_folds = []
    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BertScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(1, 11):
        augmented_train_folds = copy.deepcopy(train_folds)
        augmentation_step: augment.AugmentationStep = augment.Trafo5Step(p=i/10)  # adapt

        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)
            train_fold = copy.deepcopy(train_folds[j])
            train_fold.extend(train_folds[j])
            doubled_train_folds.append(train_fold)

        # actual training
        f_1_score = run_experiment("Experiment 5.1", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bert(unaug_train_folds=doubled_train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/trafo5/exp5.1/{names[k]}_{i/10}.json", indent=4)

    df_complete.index = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    df_complete.to_json(path_or_buf=f"./experiment_results/trafo5/exp5.1/{names[0]}.json", indent=4)


def experiment82_1():  # Probability of replacement
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bert"]
    doubled_train_folds = []
    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BertScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(1, 11):
        augmented_train_folds = copy.deepcopy(train_folds)
        augmentation_step: augment.AugmentationStep = augment.Trafo82Step(p=i/10)  # adapt

        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)
            train_fold = copy.deepcopy(train_folds[j])
            train_fold.extend(train_folds[j])
            doubled_train_folds.append(train_fold)

        # actual training
        f_1_score = run_experiment("Experiment 82.1", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bert(unaug_train_folds=doubled_train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/trafo82/exp82.1/{names[k]}_{i/10}.json", indent=4)

    df_complete.index = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    df_complete.to_json(path_or_buf=f"./experiment_results/trafo82/exp82.1/{names[0]}.json", indent=4)


def experiment82_2():  # Probability of replacement
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bert"]
    doubled_train_folds = []
    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BertScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(3):
        if i == 0:
            short_to_long=True
            long_to_short=True
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Trafo82Step(short_to_long=short_to_long,
                                                                              long_to_short=long_to_short)  # adapt
        elif i == 1:
            short_to_long = True
            long_to_short = False
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Trafo82Step(short_to_long=short_to_long,
                                                                              long_to_short=long_to_short)  # adapt
        else:
            short_to_long = False
            long_to_short = True
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Trafo82Step(short_to_long=short_to_long,
                                                                              long_to_short=long_to_short)  # adapt
        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)
            train_fold = copy.deepcopy(train_folds[j])
            train_fold.extend(train_folds[j])
            doubled_train_folds.append(train_fold)

        # actual training
        f_1_score = run_experiment("Experiment 82.2", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bert(unaug_train_folds=doubled_train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/trafo82/exp82.2/{names[k]}_{short_to_long}{long_to_short}.json", indent=4)

    df_complete.index = ["Both", "Short to Long", "Long to Short"]
    df_complete.to_json(path_or_buf=f"./experiment_results/trafo82/exp82.2/{names[0]}.json", indent=4)


def experiment100_1():  # Probability of replacement
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bert"]
    doubled_train_folds = []
    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BertScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(1, 11):
        augmented_train_folds = copy.deepcopy(train_folds)
        augmentation_step: augment.AugmentationStep = augment.Trafo100Step(prob=i/10)  # adapt

        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)
            train_fold = copy.deepcopy(train_folds[j])
            train_fold.extend(train_folds[j])
            doubled_train_folds.append(train_fold)

        # actual training
        f_1_score = run_experiment("Experiment 100.1", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bert(unaug_train_folds=doubled_train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/trafo100/exp100.1/{names[k]}_{i/10}.json", indent=4)

    df_complete.index = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    df_complete.to_json(path_or_buf=f"./experiment_results/trafo100/exp100.1/{names[0]}.json", indent=4)


def experiment100_2():  # Probability of replacement
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bert"]
    doubled_train_folds = []
    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BertScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(2):
        if i == 0:
            pos_type = True
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Trafo100Step(pos_type=pos_type)  # adapt
        else:
            pos_type = False
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Trafo100Step(pos_type=pos_type)  # adapt

        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)
            train_fold = copy.deepcopy(train_folds[j])
            train_fold.extend(train_folds[j])
            doubled_train_folds.append(train_fold)

        # actual training
        f_1_score = run_experiment("Experiment 100.2", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bert(unaug_train_folds=doubled_train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/trafo100/exp100.2/{names[k]}_{pos_type}.json", indent=4)

    df_complete.index = ["Nomen", "Adjektive"]
    df_complete.to_json(path_or_buf=f"./experiment_results/trafo100/exp100.2/{names[0]}.json", indent=4)


def experiment9_1():  # delete all sentences with length < i
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bert"]
    doubled_train_folds = []
    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BertScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(3, 13):
        augmented_train_folds = copy.deepcopy(train_folds)
        augmentation_step: augment.AugmentationStep = augment.Filter9Step(length=i)  # adapt

        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)
            train_fold = copy.deepcopy(train_folds[j])
            train_fold.extend(train_folds[j])
            doubled_train_folds.append(train_fold)

        # actual training
        f_1_score = run_experiment("Experiment 9.1", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bert(unaug_train_folds=doubled_train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/filter9/exp9.1/{names[k]}_{i}.json", indent=4)

    df_complete.index = ["3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
    df_complete.to_json(path_or_buf=f"./experiment_results/filter9/exp9.1/{names[0]}.json", indent=4)


def experiment9_2():  # test different operators
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bert"]
    doubled_train_folds = []
    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BertScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(5):
        if i == 0:
            op = ">"
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Filter9Step(op=op)  # adapt
        elif i == 1:
            op = "<"
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Filter9Step(op=op)  # adapt
        elif i == 2:
            op = ">="
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Filter9Step(op=op)  # adapt
        elif i == 3:
            op = "<="
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Filter9Step(op=op)  # adapt
        else:
            op = "=="
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Filter9Step(op=op)  # adapt
        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)
            train_fold = copy.deepcopy(train_folds[j])
            train_fold.extend(train_folds[j])
            doubled_train_folds.append(train_fold)

        # actual training
        f_1_score = run_experiment("Experiment 9.2", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bert(unaug_train_folds=doubled_train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/filter9/exp9.2/{names[k]}_{op}.json", indent=4)

    df_complete.index = [">", "<", ">=", "<=", "=="]
    df_complete.to_json(path_or_buf=f"./experiment_results/filter9/exp9.2/{names[0]}.json", indent=4)


def experiment10_1():  # delete all sentences with Activity Bio Tag Count < i
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bert"]
    doubled_train_folds = []
    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BertScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(3, 12):
        augmented_train_folds = copy.deepcopy(train_folds)
        augmentation_step: augment.AugmentationStep = augment.Filter10Step(length=i)  # adapt

        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)
            train_fold = copy.deepcopy(train_folds[j])
            train_fold.extend(train_folds[j])
            doubled_train_folds.append(train_fold)

        # actual training
        f_1_score = run_experiment("Experiment 10.1", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bert(unaug_train_folds=doubled_train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/filter10/exp10.1/{names[k]}_{i}.json", indent=4)

    df_complete.index = ["3", "4", "5", "6", "7", "8", "9", "10", "11"]
    df_complete.to_json(path_or_buf=f"./experiment_results/filter10/exp10.1/{names[0]}.json", indent=4)


def experiment10_2():  # test different operators with activity and count 3
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bert"]
    doubled_train_folds = []
    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BertScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(5):
        if i == 0:
            op = ">"
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Filter10Step(op=op)  # adapt
        elif i == 1:
            op = "<"
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Filter10Step(op=op)  # adapt
        elif i == 2:
            op = ">="
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Filter10Step(op=op)  # adapt
        elif i == 3:
            op = "<="
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Filter10Step(op=op)  # adapt
        else:
            op = "=="
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Filter10Step(op=op)  # adapt
        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)
            train_fold = copy.deepcopy(train_folds[j])
            train_fold.extend(train_folds[j])
            doubled_train_folds.append(train_fold)

        # actual training
        f_1_score = run_experiment("Experiment 10.2", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bert(unaug_train_folds=doubled_train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/filter10/exp10.2/{names[k]}_{op}.json", indent=4)

    df_complete.index = [">", "<", ">=", "<=", "=="]
    df_complete.to_json(path_or_buf=f"./experiment_results/filter10/exp10.2/{names[0]}.json", indent=4)


def experiment10_3():  # test different entitity types with "<" and count 3
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bert"]
    doubled_train_folds = []
    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BertScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(7):
        if i == 0:
            ent = "Actor"
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Filter10Step(bio=ent)  # adapt
        elif i == 1:
            ent = "Activity"
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Filter10Step(bio=ent)  # adapt
        elif i == 2:
            ent = "Activity Data"
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Filter10Step(bio=ent)  # adapt
        elif i == 3:
            ent = "Further Specification"
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Filter10Step(bio=ent)  # adapt
        elif i == 4:
            ent = "XOR Gateway"
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Filter10Step(bio=ent)  # adapt
        elif i == 5:
            ent = "Condition Specification"
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Filter10Step(bio=ent)  #
        else:
            ent = "AND Gateway"
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Filter10Step(bio=ent)  # adapt
        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)
            train_fold = copy.deepcopy(train_folds[j])
            train_fold.extend(train_folds[j])
            doubled_train_folds.append(train_fold)

        # actual training
        f_1_score = run_experiment("Experiment 10.3", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bert(unaug_train_folds=doubled_train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/filter10/exp10.3/{names[k]}_{ent}.json", indent=4)

    df_complete.index = ["Actor", "Activity", "Activity Data", "Further Specification", "XOR Gateway",
                         "Condition Specification", "AND Gateway"]
    df_complete.to_json(path_or_buf=f"./experiment_results/filter10/exp10.3/{names[0]}.json", indent=4)


def experiment19_1():  # delete all sentences with Activity Bio Tag Count < i
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bert"]
    doubled_train_folds = []
    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BertScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(3, 12):
        augmented_train_folds = copy.deepcopy(train_folds)
        augmentation_step: augment.AugmentationStep = augment.Filter19Step(length=i)  # adapt

        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)
            train_fold = copy.deepcopy(train_folds[j])
            train_fold.extend(train_folds[j])
            doubled_train_folds.append(train_fold)

        # actual training
        f_1_score = run_experiment("Experiment 19.1", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bert(unaug_train_folds=doubled_train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/filter19/exp19.1/{names[k]}_{i}.json", indent=4)

    df_complete.index = ["3", "4", "5", "6", "7", "8", "9", "10", "11"]
    df_complete.to_json(path_or_buf=f"./experiment_results/filter19/exp19.1/{names[0]}.json", indent=4)


def experiment19_2():  # test different operators with Verb and count 3
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bert"]
    doubled_train_folds = []
    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BertScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(5):
        if i == 0:
            op = ">"
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Filter19Step(op=op)  # adapt
        elif i == 1:
            op = "<"
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Filter19Step(op=op)  # adapt
        elif i == 2:
            op = ">="
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Filter19Step(op=op)  # adapt
        elif i == 3:
            op = "<="
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Filter19Step(op=op)  # adapt
        else:
            op = "=="
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Filter19Step(op=op)  # adapt
        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)
            train_fold = copy.deepcopy(train_folds[j])
            train_fold.extend(train_folds[j])
            doubled_train_folds.append(train_fold)

        # actual training
        f_1_score = run_experiment("Experiment 19.2", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bert(unaug_train_folds=doubled_train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/filter19/exp19.2/{names[k]}_{op}.json", indent=4)

    df_complete.index = [">", "<", ">=", "<=", "=="]
    df_complete.to_json(path_or_buf=f"./experiment_results/filter19/exp19.2/{names[0]}.json", indent=4)


def experiment19_3():  # test different Pos Types with "<" and count 3
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bert"]
    doubled_train_folds = []
    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BertScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(7):
        if i == 0:
            pos = "N"
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Filter19Step(pos=pos)  # adapt
        elif i == 1:
            pos = "A"
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Filter19Step(pos=pos)  # adapt
        else:
            pos = "V"
            augmented_train_folds = copy.deepcopy(train_folds)
            augmentation_step: augment.AugmentationStep = augment.Filter19Step(pos=pos)  # adapt

        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)
            train_fold = copy.deepcopy(train_folds[j])
            train_fold.extend(train_folds[j])
            doubled_train_folds.append(train_fold)

        # actual training
        f_1_score = run_experiment("Experiment 19.3", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bert(unaug_train_folds=doubled_train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/filter19/exp19.3/{names[k]}_{pos}.json", indent=4)

    df_complete.index = ["Nomen", "Adjektive", "Verben"]
    df_complete.to_json(path_or_buf=f"./experiment_results/filter19/exp19.3/{names[0]}.json", indent=4)


#######################################################
#######################################################
#######################LEONIE##########################
#######################################################
#######################################################


def experiment3_1():
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bleu"]

    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BleuScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(1, 11):
        augmented_train_folds = copy.deepcopy(train_folds)
        augmentation_step: augment.AugmentationStep = augment.Trafo3Step(no_dupl=False, max_adj=10, prob=i/10)

        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)

        # actual training
        f_1_score = run_experiment("Experiment 3.1", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bleu(unaug_train_folds=train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/trafo3/exp3.1/{names[k]}_{i/10}.json", indent=4)

    df_complete.index = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    df_complete.to_json(path_or_buf=f"./experiment_results/trafo3/exp3.1/{names[0]}.json", indent=4)


def experiment3_2():
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bleu"]

    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BleuScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(1, 11):
        augmented_train_folds = copy.deepcopy(train_folds)
        augmentation_step: augment.AugmentationStep = augment.Trafo3Step(prob=0.5, no_dupl=False, max_adj=i)

        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)

        # actual training
        f_1_score = run_experiment("Experiment 3.2", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bleu(unaug_train_folds=train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/trafo3/exp3.2/{names[k]}_{i}.json", indent=4)

    df_complete.index = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    df_complete.to_json(path_or_buf=f"./experiment_results/trafo3/exp3.2/{names[0]}.json", indent=4)


def experiment3_3():
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bleu"]

    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BleuScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(2):
        augmented_train_folds = copy.deepcopy(train_folds)
        if i == 0:
            dupl = True
            augmentation_step: augment.AugmentationStep = augment.Trafo3Step(no_dupl=True, max_adj=10, prob=0.5)
        else:
            dupl = False
            augmentation_step: augment.AugmentationStep = augment.Trafo3Step(no_dupl=False, max_adj=10, prob=0.5)

        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)

        # actual training
        f_1_score = run_experiment("Experiment 3.3", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bleu(unaug_train_folds=train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/trafo3/exp3.3/{names[k]}_{dupl}.json", indent=4)

    df_complete.index = ["True", "False"]
    df_complete.to_json(path_or_buf=f"./experiment_results/trafo3/exp3.3/{names[0]}.json", indent=4)


def experiment39_1():
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bleu"]

    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BleuScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(1, 11):
        augmented_train_folds = copy.deepcopy(train_folds)
        augmentation_step: augment.AugmentationStep = augment.Trafo39Step(prob=i/10)

        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)

        # actual training
        f_1_score = run_experiment("Experiment 39.1", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bleu(unaug_train_folds=train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/trafo39/exp39.1/{names[k]}_{i/10}.json", indent=4)

    df_complete.index = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    df_complete.to_json(path_or_buf=f"./experiment_results/trafo39/exp39.1/{names[0]}.json", indent=4)


def experiment86_1():
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bleu"]

    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BleuScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(1, 11):
        augmented_train_folds = copy.deepcopy(train_folds)
        augmentation_step: augment.AugmentationStep = augment.Trafo86Step(no_dupl=False, prob=i/10, max_noun=10, kind_of_replace=2)

        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)

        # actual training
        f_1_score = run_experiment("Experiment 86.1", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bleu(unaug_train_folds=train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/trafo86/exp86.1/{names[k]}_{i/10}.json", indent=4)

    df_complete.index = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    df_complete.to_json(path_or_buf=f"./experiment_results/trafo86/exp86.1/{names[0]}.json", indent=4)


def experiment86_2():
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bleu"]

    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BleuScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(1, 11):
        augmented_train_folds = copy.deepcopy(train_folds)
        augmentation_step: augment.AugmentationStep = augment.Trafo86Step(no_dupl=False, prob=0.5, max_noun=i, kind_of_replace=2)

        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)

        # actual training
        f_1_score = run_experiment("Experiment 86.2", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bleu(unaug_train_folds=train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/trafo86/exp86.2/{names[k]}_{i}.json", indent=4)

    df_complete.index = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    df_complete.to_json(path_or_buf=f"./experiment_results/trafo86/exp86.2/{names[0]}.json", indent=4)


def experiment86_3():
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bleu"]

    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BleuScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(3):
        augmented_train_folds = copy.deepcopy(train_folds)
        augmentation_step: augment.AugmentationStep = augment.Trafo86Step(no_dupl=False, prob=0.5, max_noun=10, kind_of_replace=i)

        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)

        # actual training
        f_1_score = run_experiment("Experiment 86.3", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bleu(unaug_train_folds=train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        if i == 0:
            kind = "hyponym"
        elif i == 1:
            kind = "hypernym"
        else:
            kind = "random"

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/trafo86/exp86.3/{names[k]}_{kind}.json", indent=4)

    df_complete.index = ["hyponym", "hypernym", "random"]
    df_complete.to_json(path_or_buf=f"./experiment_results/trafo86/exp86.3/{names[0]}.json", indent=4)


def experiment86_4():
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bleu"]

    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BleuScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(2):
        augmented_train_folds = copy.deepcopy(train_folds)
        if i == 0:
            dupl = True
            augmentation_step: augment.AugmentationStep = augment.Trafo86Step(no_dupl=True, prob=0.5, max_noun=10, kind_of_replace=2)
        else:
            dupl = False
            augmentation_step: augment.AugmentationStep = augment.Trafo86Step(no_dupl=False, prob=0.5, max_noun=10, kind_of_replace=2)

        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)

        # actual training
        f_1_score = run_experiment("Experiment 86.4", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bleu(unaug_train_folds=train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/trafo86/exp86.4/{names[k]}_{dupl}.json", indent=4)

    df_complete.index = ["True", "False"]
    df_complete.to_json(path_or_buf=f"./experiment_results/trafo86/exp86.4/{names[0]}.json", indent=4)


def experiment90_1():
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bleu"]

    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BleuScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(1, 11):
        augmented_train_folds = copy.deepcopy(train_folds)
        augmentation_step: augment.AugmentationStep = augment.Trafo90Step(prob=i/10)

        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)

        # actual training
        f_1_score = run_experiment("Experiment 90.1", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bleu(unaug_train_folds=train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/trafo90/exp90.1/{names[k]}_{i/10}.json", indent=4)

    df_complete.index = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    df_complete.to_json(path_or_buf=f"./experiment_results/trafo90/exp90.1/{names[0]}.json", indent=4)


def experiment103_1():
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bleu"]

    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BleuScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(1, 11):
        augmented_train_folds = copy.deepcopy(train_folds)
        augmentation_step: augment.AugmentationStep = augment.Trafo103Step(prob=i/10, num_of_words=2)

        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)

        # actual training
        f_1_score = run_experiment("Experiment 103.1", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bleu(unaug_train_folds=train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/trafo103/exp103.1/{names[k]}_{i/10}.json", indent=4)

    df_complete.index = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    df_complete.to_json(path_or_buf=f"./experiment_results/trafo103/exp103.1/{names[0]}.json", indent=4)


def experiment103_2():
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bleu"]

    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BleuScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(1, 11):
        augmented_train_folds = copy.deepcopy(train_folds)
        augmentation_step: augment.AugmentationStep = augment.Trafo103Step(prob=0.5, num_of_words=i)

        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)

        # actual training
        f_1_score = run_experiment("Experiment 103.2", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bleu(unaug_train_folds=train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/trafo103/exp103.2/{names[k]}_{i}.json", indent=4)

    df_complete.index = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    df_complete.to_json(path_or_buf=f"./experiment_results/trafo103/exp103.2/{names[0]}.json", indent=4)


def experiment103_3():
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bleu"]

    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BleuScore$'])

    # augment the dataset - for i in range of the parameter
    ent_list = ['JJ', 'NN', 'NNS', 'NNP', 'RB', 'DT', 'IN', 'VBN', 'VBP', 'VBZ', 'PRP', 'WP']
    for i in range(12):
        str = [ent_list[i]]
        str_name = ent_list[i]
        augmented_train_folds = copy.deepcopy(train_folds)
        augmentation_step: augment.AugmentationStep = augment.Trafo103Step(prob=0.5, num_of_words=2, kind_of_word=str)

        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)

        # actual training
        f_1_score = run_experiment("Experiment 103.3", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bleu(unaug_train_folds=train_folds,
                                              aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/trafo103/exp103.3/{names[k]}_{str_name}.json", indent=4)

    df_complete.index = ['JJ', 'NN', 'NNS', 'NNP', 'RB', 'DT', 'IN', 'VBN', 'VBP', 'VBZ', 'PRP', 'WP']
    df_complete.to_json(path_or_buf=f"./experiment_results/trafo103/exp103.3/{names[0]}.json", indent=4)


def experiment40_1():
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bleu"]

    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BleuScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(1, 11):
        augmented_train_folds = copy.deepcopy(train_folds)
        augmentation_step: augment.AugmentationStep = augment.Trafo40Step(prob=i/10)

        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)

        # actual training
        f_1_score = run_experiment("Experiment 40.1", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bleu(unaug_train_folds=train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/trafo40/exp40.1/{names[k]}_{i/10}.json", indent=4)

    df_complete.index = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    df_complete.to_json(path_or_buf=f"./experiment_results/trafo40/exp40.1/{names[0]}.json", indent=4)


def experiment40_2():
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bleu"]

    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BleuScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(6):
        augmented_train_folds = copy.deepcopy(train_folds)
        if i == 0:
            str = "speaker"
            augmentation_step: augment.AugmentationStep = augment.Trafo40Step(prob=0.5, speaker_p=True, filler_p=False,
                                                                              uncertain_p=False)
        elif i == 1:
            str = "filler"
            augmentation_step: augment.AugmentationStep = augment.Trafo40Step(prob=0.5, speaker_p=False, filler_p=True,
                                                                              uncertain_p=False)
        elif i == 2:
            str = "uncertain"
            augmentation_step: augment.AugmentationStep = augment.Trafo40Step(prob=0.5, speaker_p=False, filler_p=False,
                                                                              uncertain_p=True)
        elif i == 3:
            str = "speaker.filler"
            augmentation_step: augment.AugmentationStep = augment.Trafo40Step(prob=0.5, speaker_p=True, filler_p=True,
                                                                              uncertain_p=False)
        elif i == 4:
            str = "speaker.uncertain"
            augmentation_step: augment.AugmentationStep = augment.Trafo40Step(prob=0.5, speaker_p=True, filler_p=False,
                                                                              uncertain_p=True)
        else:
            str = "filler.uncertain"
            augmentation_step: augment.AugmentationStep = augment.Trafo40Step(prob=0.5, speaker_p=False, filler_p=True,
                                                                              uncertain_p=True)

        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)

        # actual training
        f_1_score = run_experiment("Experiment 40.2", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bleu(unaug_train_folds=train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/trafo40/exp40.2/{names[k]}_{str}.json", indent=4)

    df_complete.index = ["speaker", "filler", "uncertain", "speaker&filler", "speaker&uncertain", "filler&uncertain"]
    df_complete.to_json(path_or_buf=f"./experiment_results/trafo40/exp40.2/{names[0]}.json", indent=4)


def experiment40_3():
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bleu"]

    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BleuScore$'])

    # augment the dataset - for i in range of the parameter
    ent_list = ['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                   'Condition Specification', 'AND Gateway', 'All']
    for i in range(8):
        str = [ent_list[i]]
        str_name = ent_list[i]
        augmented_train_folds = copy.deepcopy(train_folds)
        augmentation_step: augment.AugmentationStep = augment.Trafo40Step(prob=0.5, tags=str)

        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)

        # actual training
        f_1_score = run_experiment("Experiment 40.3", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bleu(unaug_train_folds=train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/trafo40/exp40.3/{names[k]}_{str_name}.json", indent=4)

    df_complete.index = ['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                   'Condition Specification', 'AND Gateway', 'All']
    df_complete.to_json(path_or_buf=f"./experiment_results/trafo40/exp40.3/{names[0]}.json", indent=4)


def experiment40_4():
    # Get the data for augmenting and training
    train_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'jsonl/fold_{i}/test.json') for i in range(5)]
    names = ["all_means", "ttr", "ucer", "ttr_mean", "ucer_mean", "bleu"]

    # specific for this experiment
    df_complete: pd.DataFrame = pd.DataFrame(columns=['$F_{1}$', '$TTR$', '$UCER$', '$BleuScore$'])

    # augment the dataset - for i in range of the parameter
    for i in range(10):
        augmented_train_folds = copy.deepcopy(train_folds)
        if i == 0:
            str = "uncertain&Further"
            augmentation_step: augment.AugmentationStep = augment.Trafo40Step(prob=0.5, speaker_p=False, filler_p=False,
                                                                              uncertain_p=True, tags=['Further Specification'])
        elif i == 1:
            str = "uncertain&XOR"
            augmentation_step: augment.AugmentationStep = augment.Trafo40Step(prob=0.5, speaker_p=False, filler_p=False,
                                                                              uncertain_p=True, tags=["XOR Gateway"])
        elif i == 2:
            str = "uncertain&AND"
            augmentation_step: augment.AugmentationStep = augment.Trafo40Step(prob=0.5, speaker_p=False, filler_p=False,
                                                                              uncertain_p=True, tags=['AND Gateway'])
        elif i == 3:
            str = "uncertain&Condition"
            augmentation_step: augment.AugmentationStep = augment.Trafo40Step(prob=0.5, speaker_p=False, filler_p=False,
                                                                              uncertain_p=True, tags=['Condition Specification'])
        elif i == 4:
            str = "uncertain&Data"
            augmentation_step: augment.AugmentationStep = augment.Trafo40Step(prob=0.5, speaker_p=False, filler_p=False,
                                                                              uncertain_p=True, tags=['Activity Data'])
        elif i == 5:
            str = "speaker&Further"
            augmentation_step: augment.AugmentationStep = augment.Trafo40Step(prob=0.5, speaker_p=True,
                                                                              filler_p=False,
                                                                              uncertain_p=False,
                                                                              tags=['Further Specification'])
        elif i == 6:
            str = "speaker&XOR"
            augmentation_step: augment.AugmentationStep = augment.Trafo40Step(prob=0.5, speaker_p=True,
                                                                              filler_p=False,
                                                                              uncertain_p=False,
                                                                              tags=["XOR Gateway"])
        elif i == 7:
            str = "speaker&AND"
            augmentation_step: augment.AugmentationStep = augment.Trafo40Step(prob=0.5, speaker_p=True,
                                                                              filler_p=False,
                                                                              uncertain_p=False,
                                                                              tags=['AND Gateway'])
        elif i == 8:
            str = "speaker&Condition"
            augmentation_step: augment.AugmentationStep = augment.Trafo40Step(prob=0.5, speaker_p=True,
                                                                              filler_p=False,
                                                                              uncertain_p=False,
                                                                              tags=['Condition Specification'])
        else:
            str = "speaker&Data"
            augmentation_step: augment.AugmentationStep = augment.Trafo40Step(prob=0.5, speaker_p=True,
                                                                              filler_p=False,
                                                                              uncertain_p=False,
                                                                              tags=['Activity Data'])

        # actual augmentation
        for j in range(5):
            augmented_train_set = augment.run_augmentation(augmented_train_folds[j], augmentation_step)
            augmented_train_set.extend(train_folds[j])
            augmented_train_folds[j] = copy.deepcopy(augmented_train_set)

        # actual training
        f_1_score = run_experiment("Experiment 40.4", augmented_train_folds, test_folds)

        # evaluation
        all_scores = evaluate_experiment_bleu(unaug_train_folds=train_folds,
                            aug_train_folds=augmented_train_folds, f_score=f_1_score)

        df_complete = df_complete.append(all_scores[0], ignore_index=True)

        for k in range(1, len(all_scores) - 1):
            df = all_scores[k]
            df.to_json(path_or_buf=f"./experiment_results/trafo40/exp40.4/{names[k]}_{str}.json", indent=4)

    df_complete.index = ["uncertain&Further", "uncertain&XOR", "uncertain&AND", "uncertain&Condition", "uncertain&Data",
                         "speaker&Further", "speaker&XOR", "speaker&AND", "speaker&Condition", "speaker&Data"]
    df_complete.to_json(path_or_buf=f"./experiment_results/trafo40/exp40.4/{names[0]}.json", indent=4)


#experiment1_1()
#experiment101_1()
#experiment3_1()
#experiment3_2()
#experiment3_3()
#experiment39_1()
#experiment86_1()
#experiment86_2()
#experiment86_3()
#experiment86_4()
experiment33_1()