import copy

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
class Plot:
    def __init__(self):
        pass

    @staticmethod
    def plot_scatter_with_regression(path):
        palette = sns.color_palette("flare", n_colors=20)
        df = pd.read_json(path_or_buf=path)
        #prob_list = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
        prob_list = ["0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5", "0.55", "0.6",
                             "0.65", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95", "1.0"]
        print(df)
        df2 = copy.deepcopy(df)
        df2["Prob"] = prob_list
        print(df2)
        ttr = []
        ucer = []
        bleu = []
        for i in range(1, 21):
            ttr.append([df["$F_{1}$"][i - 1], df["$TTR$"][i - 1], f"{i/20}"])
            ucer.append([df["$F_{1}$"][i - 1], df["$UCER$"][i - 1], f"{i/20}"])
        df_ttr = pd.DataFrame(ttr, columns=['F1 Score', 'TTR', 'Prob'])
        df_ucer = pd.DataFrame(ucer, columns=['F1 Score', 'UCER', 'Prob'])
        #fig = sns.lmplot(x='TTR', y='F1 Score', data=df_ttr,  palette=palette, hue='Prob', fit_reg=True,)
        fig = sns.lineplot(x='$TTR$', y='$F_{1}$', data=df2,  palette=palette)
        #sns.lmplot(x='UCER', y='F1 Score', data=df_ucer, fit_reg=True)
        #fig.savefig("./../experiment_results/trafo3/exp3.1/all_means.png")
        #fig.savefig("./../experiment_results/trafo3/exp3.1/all_means.pdf")
df2 = pd.DataFrame([[0.6, 0.4], [0.65, 0.45], [0.7, 0.6],

                   [0.75, 0.65], [0.9, 0.75]],

                  columns=['F1 Score', 'TTR'])
path = "./../experiment_results/trafo3/exp3.1/all_means.json"
columns=['F1 Score', 'TTR']

prob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
f1 = [0.49, 0.51, 0.52, 0.53, 0.55, 0.58, 0.59, 0.6, 0.61, 0.63]
bleu = [0.99, 0.975, 0.97, 0.965, 0.96, 0.955, 0.954, 0.95, 0.949, 0.94]
bert = [0.8, 0.809, 0.83, 0.84, 0.8456, 0.86, 0.87345, 0.88, 0.889, 0.9]
df4 = pd.DataFrame()
sns.set_theme()
df4["F1"] = f1
df4["Probability"] = prob
df4["Bert Score"] = bert
palette = sns.color_palette("flare", n_colors=10)
fig = sns.lineplot(x='Bert Score', y='F1', data=df4)
figg = fig.figure
figg.savefig("./../experiment_results/trafo3/exp3.1/test.png")
#Plot.plot_scatter_with_regression(path)
plt.show()
#Plot.plot_scatter_with_regression(path)
plt.show()