import os

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import seaborn as sns

import data
import main
import pipeline


def build_data():
    train_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/test.json') for i in range(5)]

    dataframe: pd.DataFrame
    if os.path.isfile('negative_sampling_rate_data.pkl'):
        print('continue with building data')
        dataframe = pandas.read_pickle('negative_sampling_rate_data.pkl')
    else:
        print('starting new data building step')
        dataframe = pd.DataFrame(columns=['negative_sampling_rate', '$P$', '$R$', '$F_{1}$'])

    index = 0
    for negative_sampling_rate in np.linspace(0.0, 200.0, num=201):
        index += 1
        if negative_sampling_rate in dataframe['negative_sampling_rate'].values:
            print(f'rate {negative_sampling_rate} already in dataframe, continuing')
            continue

        print(f'building scores for rate {negative_sampling_rate}')
        res = main.cross_validate_pipeline(
            p=pipeline.Pipeline(name='negative sampling optimization', steps=[
                pipeline.CatBoostRelationExtractionStep(name='perfect entities', context_size=2,
                                                        num_trees=100, negative_sampling_rate=negative_sampling_rate)
            ]),
            train_folds=train_folds,
            test_folds=test_folds
        )

        scores = list(res.values())[0]

        dataframe = dataframe.append(
            {'negative_sampling_rate': negative_sampling_rate,
             '$P$': scores.overall_scores.p,
             '$R$': scores.overall_scores.r,
             '$F_{1}$': scores.overall_scores.f1},
            ignore_index=True)

        pandas.to_pickle(dataframe, 'negative_sampling_rate_data.pkl')


def plot_data(name: str, fig_size=(6.4, 4.8)):
    df: pd.DataFrame = pandas.read_pickle('negative_sampling_rate_data.pkl')
    df = df.set_index('negative_sampling_rate')
    print(df)

    sns.set_theme()
    df.plot(kind='line', style=['-.', '--', '-'], figsize=fig_size,
            xlabel='negative sampling rate $r_n$', ylabel='value')
    plt.tight_layout()

    os.makedirs('figures/hyper-opt', exist_ok=True)
    plt.savefig(f'figures/hyper-opt/{name}.png')
    plt.savefig(f'figures/hyper-opt/{name}.pdf')


if __name__ == '__main__':
    build_data()
    plot_data('p-vs-r-vs-f1', fig_size=(6.4, 2.2))
