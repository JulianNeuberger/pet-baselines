import pandas as pd

df: pd.DataFrame = pd.read_pickle('experiments.jerex.pkl')
df = df.reset_index()
df = df.loc[df['tag'] == 'overall']
df = df[['experiment_name', 'f1', 'p', 'r']]
df = df.melt(id_vars=['experiment_name'], var_name='metric', value_name='score')
print(df.sort_values(by=['experiment_name', 'metric']))
