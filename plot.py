import os

import numpy as np
import pandas
import pandas as pd
import seaborn as sns
import sqlite3

import matplotlib.pyplot as plt

# Read sqlite query results into a pandas DataFrame
con = sqlite3.connect('optuna.db')
df: pd.DataFrame
df = pd.read_sql_query('SELECT t.trial_id, value, param_name, param_value '
                       'FROM trial_values '
                       'LEFT JOIN trials t ON t.trial_id = trial_values.trial_id and study_id=1 '
                       'LEFT JOIN trial_params tp on t.trial_id = tp.trial_id', con)

trial_value_df: pandas.DataFrame = df.groupby('trial_id').first()['value']
parameter_df = df.pivot_table(index='trial_id', columns='param_name', values='param_value', aggfunc='first')

df = parameter_df.join(trial_value_df, on='trial_id')
df = df.pivot_table(index='cluster_overlap', columns='mention_overlap', values='value')
df = df[[c for c in df.columns if float(c) > 0]]

print(df)

x = df.index.values
y = df.columns.values
z = df.values

X, Y = np.meshgrid(x, y)
Z = df.T

sns.set_theme()

levels = np.linspace(.2, .5, num=15)

cs = plt.contourf(X, Y, Z, levels=levels)
plt.xlabel(r'$\alpha_c$')
plt.ylabel(r'$\alpha_m$')
ticks = [f'{l:.1%}' for l in levels]
cbar = plt.colorbar(cs, ticks=levels)
cbar.ax.set_xticklabels(ticks)
cbar.ax.set_ylabel('F1')

os.makedirs(os.path.join('figures', 'hyper-opt'), exist_ok=True)
plt.savefig(os.path.join('figures', 'hyper-opt', 'co-ref-params.pdf'))
plt.savefig(os.path.join('figures', 'hyper-opt', 'co-ref-params.png'))
