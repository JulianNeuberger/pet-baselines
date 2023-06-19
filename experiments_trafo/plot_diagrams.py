import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
class Plot:
    def __init__(self):
        pass

    @staticmethod
    def plot_scatter_with_regression(path):
        palette = sns.color_palette("hls", 8)
        df = pd.read_json(path_or_buf=path)
        prob_list = ["0.2", "0.4", "0.6", "0.8", "1.0"]
        #print(df)
        ttr = []
        ucer = []
        bleu = []
        for i in range(5):
            ttr.append([df["$F_{1}$"][i], df["$TTR$"][i], f"{i/5}"])
            ucer.append([df["$F_{1}$"][i], df["$UCER$"][i], f"{i/5}"])
        df_ttr = pd.DataFrame(ttr, columns=['F1 Score', 'TTR', 'Prob'])
        df_ucer = pd.DataFrame(ucer, columns=['F1 Score', 'UCER', 'Prob'])
        sns.lmplot(x='F1 Score', y='TTR', data=df_ttr,  palette=palette, hue='Prob', fit_reg=True,)
        sns.lmplot(x='F1 Score', y='UCER', data=df_ucer, fit_reg=True)

df2 = pd.DataFrame([[0.6, 0.4], [0.65, 0.45], [0.7, 0.6],

                   [0.75, 0.65], [0.9, 0.75]],

                  columns=['F1 Score', 'TTR'])
path = "./../experiment_results/trafo3/exp3.3/all_means.json"

Plot.plot_scatter_with_regression(path)
plt.show()