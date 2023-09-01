import os
import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import seaborn as sns

import data
import main
import pipeline


def build_data(data_file_path: str,
               params: typing.Dict[str, typing.Any],
               parameter_to_tune: str,
               parameter_range: typing.Iterable):
    train_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/test.json') for i in range(5)]

    dataframe: pd.DataFrame
    if os.path.isfile(data_file_path):
        print('continue with building data')
        dataframe = pandas.read_pickle(data_file_path)
    else:
        dir_path = os.path.dirname(data_file_path)
        print(f'starting new data building step at {dir_path}')
        os.makedirs(dir_path, exist_ok=True)
        dataframe = pd.DataFrame(columns=[parameter_to_tune, '$P$', '$R$', '$F_{1}$'])

    index = 0
    for param_value in parameter_range:
        index += 1
        if param_value in dataframe[parameter_to_tune].values:
            print(f'{parameter_to_tune}={param_value} already in dataframe, continuing')
            continue

        params.update({parameter_to_tune: param_value})

        print(f'building scores for {parameter_to_tune}={param_value}')
        res = main.cross_validate_pipeline(
            p=pipeline.Pipeline(name=f'{parameter_to_tune} optimization', steps=[
                pipeline.CatBoostRelationExtractionStep(name=f'{parameter_to_tune}={param_value}', **params)
            ]),
            train_folds=train_folds,
            test_folds=test_folds
        )

        scores = list(res.values())[0]

        dataframe = dataframe.append(
            {parameter_to_tune: param_value,
             '$P$': scores.overall_scores.p,
             '$R$': scores.overall_scores.r,
             '$F_{1}$': scores.overall_scores.f1},
            ignore_index=True)

        pandas.to_pickle(dataframe, data_file_path)

        if index % 5 == 0:
            plot_data(f'{parameter_to_tune}_{index}', parameter_to_tune, data_file_path)


def plot_data(name: str, attribute_name: str, data_file_path: str,
              filter_predicate: typing.Callable[[pd.Series], bool] = None, fig_size=(6.4, 4.8)):
    df: pd.DataFrame = pandas.read_pickle(data_file_path)
    df = df.set_index(attribute_name)
    df = df.sort_index()
    if filter_predicate is not None:
        # noinspection PyTypeChecker
        df = df[df.apply(filter_predicate, axis=1)]

    sns.set_theme()
    df.plot(kind='line', style=['-.', '--', '-'], figsize=fig_size,
            xlabel=f'{attribute_name}', ylabel='value')
    plt.tight_layout()

    os.makedirs(f'figures/hyper-opt/{attribute_name}', exist_ok=True)
    plt.savefig(f'figures/hyper-opt/{attribute_name}/{name}.png')
    plt.savefig(f'figures/hyper-opt/{attribute_name}/{name}.pdf')


if __name__ == '__main__':
    attribute = 'num_passes'
    file_path = f'figures/hyper-opt/{attribute}/{attribute}.pkl'
    build_data(
        data_file_path=file_path,
        params={
            'learning_rate': None,
            'num_trees': 100,
            'negative_sampling_rate': 40,
            'context_size': 2,
            'depth': 4,
            'use_pos_features': False,
            'use_embedding_features': False,
            'class_weighting': 0.0,
            'num_passes': 1
        },
        parameter_to_tune=attribute,
        parameter_range=list(range(1, 6))
    )

    plot_data(f'{attribute}_all', attribute, file_path, fig_size=(6.4, 2.2))
