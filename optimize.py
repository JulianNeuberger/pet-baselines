import pathlib
import time

import numpy as np
import optuna

import data
import main
from eval import metrics


def runner():
    def objective(trial: optuna.Trial):
        start = time.time()
        pipeline_config = main.PipelineConfig(mention_overlap=trial.suggest_float('mention_overlap', 0.0, 1.0),
                                              cluster_overlap=trial.suggest_float('cluster_overlap', 0.0, 1.0),
                                              ner_strategy='frequency', crf_model_path=pathlib.Path())
        f1_stats = []
        for n_fold, (train_fold, test_fold) in enumerate(folds):
            input_docs = [d.copy(clear_entities=True) for d in test_fold]
            prediction = main.entity_extraction_module(pipeline_config, naive=False, documents=input_docs)
            f1_stats.append(metrics.entity_f1_stats(
                predicted_documents=prediction,
                ground_truth_documents=test_fold,
                min_num_mentions=2
            ))

        objective_value = np.mean([f1 for _, _, f1 in f1_stats])
        print(f'Ran for {time.time() - start:.4f}s and produced {objective_value:.2%} F1')
        return objective_value

    train_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/test.json') for i in range(5)]
    folds = list(zip(train_folds, test_folds))

    search_space = {
        'mention_overlap': np.linspace(0.0, 1.0, num=100),
        'cluster_overlap': np.linspace(0.0, 1.0, num=100)
    }

    study = optuna.create_study(study_name='co-ref-optimization-perfect-entities',
                                storage='sqlite:///optuna.db',
                                direction='maximize',
                                load_if_exists=True,
                                sampler=optuna.samplers.GridSampler(search_space))
    study.optimize(objective)


if __name__ == '__main__':
    runner()
