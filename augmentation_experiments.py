import random
import sys
import traceback
import typing

import optuna
import sklearn.model_selection

import augment
import data
import pipeline
from augment import (
    base,
    trafo3,
    trafo5,
    trafo6,
    trafo8,
    trafo24,
    trafo26,
    trafo39,
    trafo40,
    params,
    trafo58,
    trafo62,
    trafo79,
    trafo82,
    trafo86,
    trafo88,
    trafo90,
    trafo100,
    trafo101,
    trafo103,
    trafo106,
    trafo_null,
    trafo_insert,
)
from data import loader, model
from main import cross_validate_pipeline

strategies: typing.List[typing.Type[base.AugmentationStep]] = [
    trafo3.Trafo3Step,
    trafo5.Trafo5Step,
    trafo6.Trafo6Step,
    trafo8.Trafo8Step,  # long runtime
    trafo24.Trafo24Step,
    trafo26.Trafo26Step,
    trafo39.Trafo39Step,
    trafo40.Trafo40Step,
    trafo58.Trafo58Step,  # runs too long?
    trafo62.Trafo62Step,  # runs too long
    trafo79.Trafo79Step,
    trafo82.Trafo82Step,
    trafo86.Trafo86HyponymReplacement,
    trafo86.Trafo86HypernymReplacement,
    trafo88.Trafo88Step,
    trafo90.Trafo90Step,
    trafo100.Trafo100Step,
    trafo101.Trafo101Step,
    trafo103.Trafo103Step,
    trafo106.Trafo106Step,
    trafo_null.TrafoNullStep,
    trafo_insert.TrafoInsertStep,
    # cheating.CheatingTransformationStep,
]

max_runs_per_step = 25
random_state = 42
random.seed(random_state)


def suggest_param(param: params.Param, trial: optuna.Trial) -> typing.Any:
    if isinstance(param, params.IntegerParam):
        return trial.suggest_int(
            name=param.name, low=param.min_value, high=param.max_value
        )

    if isinstance(param, params.FloatParam):
        return trial.suggest_float(
            name=param.name, low=param.min_value, high=param.max_value
        )

    if isinstance(param, params.ChoiceParam):
        if param.max_num_picks > 1:
            choices = param.get_combinations_as_bit_masks()
            choice = trial.suggest_categorical(name=param.name, choices=choices)
            choice_as_list = param.bit_mask_to_choices(choice)
            return choice_as_list
        return trial.suggest_categorical(name=param.name, choices=param.choices)

    if isinstance(param, params.BooleanParameter):
        return trial.suggest_categorical(name=param.name, choices=[True, False])


def instantiate_step(
    step_class: typing.Type[base.AugmentationStep],
    trial: optuna.Trial,
    dataset: typing.List[model.Document],
) -> base.AugmentationStep:
    suggested_params = {
        p.name: suggest_param(p, trial) for p in step_class.get_params()
    }
    return step_class(dataset, **suggested_params)


def objective_factory(
    augmenter_class: typing.Type[base.AugmentationStep],
    pipeline_step_class: typing.Type[pipeline.PipelineStep],
    documents: typing.List[data.Document],
    **kwargs,
):
    def objective(trial: optuna.Trial):
        kf = sklearn.model_selection.KFold(
            n_splits=5, random_state=random_state, shuffle=True
        )
        augmentation_rate = trial.suggest_float("augmentation_rate", low=0.0, high=10.0)
        un_augmented_train_folds = []
        augmented_train_folds: typing.List[typing.List[data.Document]] = []
        test_folds = []
        for train_indices, test_indices in kf.split(documents):
            test_documents = [documents[i] for i in test_indices]
            un_augmented_train_documents = [documents[i] for i in train_indices]

            step = instantiate_step(
                augmenter_class, trial, un_augmented_train_documents
            )

            augmented_train_documents, _ = augment.run_augmentation(
                un_augmented_train_documents, step, augmentation_rate
            )
            random.shuffle(augmented_train_documents)
            augmented_train_folds.append(augmented_train_documents)
            un_augmented_train_folds.append(un_augmented_train_documents)
            print(
                f"Augmented {len(un_augmented_train_documents)} documents "
                f"with augmentation rate of {augmentation_rate:.4f} "
                f"resulting in {len(augmented_train_documents)} documents"
            )
            test_folds.append(test_documents)

        augmented_pipeline_step = pipeline_step_class(
            name="crf mention extraction", **kwargs
        )
        augmented_results = cross_validate_pipeline(
            p=pipeline.Pipeline(
                name=f"augmentation-{pipeline_step_class.__name__}",
                steps=[augmented_pipeline_step],
            ),
            train_folds=augmented_train_folds,
            test_folds=test_folds,
            save_results=False,
        )

        unaugmented_pipeline_step = pipeline_step_class(
            name="crf mention extraction", **kwargs
        )
        un_augmented_results = cross_validate_pipeline(
            p=pipeline.Pipeline(
                name=f"augmentation-{pipeline_step_class.__name__}",
                steps=[unaugmented_pipeline_step],
            ),
            train_folds=un_augmented_train_folds,
            test_folds=test_folds,
            save_results=False,
        )

        augmented_f1 = augmented_results[augmented_pipeline_step].overall_scores.f1
        un_augmented_f1 = un_augmented_results[
            unaugmented_pipeline_step
        ].overall_scores.f1
        improvement = augmented_f1 - un_augmented_f1
        print(f"Improvement of {improvement:.2%}")
        return improvement

    return objective


def main():
    device = "CPU"
    device_ids = None

    if len(sys.argv) > 1:
        assert len(sys.argv) == 3, (
            "If you specify devices to train on, please specify either GPU/CPU "
            "as first argument and the id(s) as second one, see "
            "https://catboost.ai/en/docs/features/training-on-gpu"
        )
        device = sys.argv[1]
        device_ids = sys.argv[2]

    for strategy_class in strategies:
        errors = []
        try:
            strategy_class.validate_params(strategy_class)
        except TypeError as e:
            errors.append(e)
        if len(errors) > 0:
            raise AssertionError("\n".join([str(e) for e in errors]))

    all_documents = loader.read_documents_from_json("./jsonl/all.jsonl")
    pipeline_step_class = pipeline.CatBoostRelationExtractionStep

    for strategy_class in strategies:
        print(f"Running optimization for strategy {strategy_class.__name__}")
        objective = objective_factory(
            strategy_class,
            pipeline_step_class,
            all_documents,
            num_trees=100,
            device=device,
            device_ids=device_ids,
        )
        study = optuna.create_study(
            direction="maximize",
            load_if_exists=True,
            study_name=f"{strategy_class.__name__}-{pipeline_step_class.__name__}",
            storage="mysql://optuna@localhost/pet_data_augment",
        )
        trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
        if len(trials) >= max_runs_per_step:
            print(
                f"All trials for {strategy_class.__name__} already ran, continuing..."
            )
            continue
        try:
            study.optimize(
                objective,
                callbacks=[
                    optuna.study.MaxTrialsCallback(
                        n_trials=max_runs_per_step,
                        states=(optuna.trial.TrialState.COMPLETE,),
                    )
                ],
            )
        except Exception as e:
            if type(e) == KeyboardInterrupt:
                raise e
            print(f"Error in strategy {strategy_class.__name__}, skipping.")
            print(traceback.format_exc())


if __name__ == "__main__":
    main()
