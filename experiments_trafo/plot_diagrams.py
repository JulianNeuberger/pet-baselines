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
            ttr.append([df["$F_{1}_CRF$"][i - 1], df["$TTR$"][i - 1], f"{i/20}"])
            ucer.append([df["$F_{1}_CRF$"][i - 1], df["$UCER$"][i - 1], f"{i/20}"])
        df_ttr = pd.DataFrame(ttr, columns=['F1 Score', 'TTR', 'Prob'])
        df_ucer = pd.DataFrame(ucer, columns=['F1 Score', 'UCER', 'Prob'])
        #fig = sns.lmplot(x='TTR', y='F1 Score', data=df_ttr,  palette=palette, hue='Prob', fit_reg=True,)
        fig = sns.lineplot(x='$TTR$', y='$F_{1}_CRF$', data=df2,  palette=palette)
        #sns.lmplot(x='UCER', y='F1 Score', data=df_ucer, fit_reg=True)
        #fig.savefig("./../experiment_results/trafo3/exp3.1/all_means.png")
        #fig.savefig("./../experiment_results/trafo3/exp3.1/all_means.pdf")

    @staticmethod
    def all_means_prob_bert():
        sns.set_theme()
        str = "101"
        str2 = "101.1"
        path = f"./../experiment_results/trafo{str}/exp{str2}/all_means.json"
        df = pd.read_json(path_or_buf=path)

        prob_list = ["0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5", "0.55", "0.6",
                     "0.65", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95", "1.0"]

        df["Probability"] = prob_list
        fig1 = sns.lineplot(x='Probability', y='F1 CRF', data=df)
        figg1 = fig1.figure
        figg1.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_means_prob_f1.png")
        figg1.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_means_prob_f1.pdf")
        plt.figure()

        fig2 = sns.lineplot(x='TTR', y='F1 CRF', data=df)
        figg2 = fig2.figure
        figg2.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_means_ttr_f1.png")
        figg2.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_means_ttr_f1.pdf")
        plt.figure()

        fig3 = sns.lineplot(x='Probability', y='TTR', data=df)
        figg3 = fig3.figure
        figg3.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_means_ttr_prob.png")
        figg3.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_means_ttr_prob.pdf")
        plt.figure()

        fig4 = sns.lineplot(x='UCER', y='F1 CRF', data=df)
        figg4 = fig4.figure
        figg4.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_means_ucer_f1.png")
        figg4.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_means_ucer_f1.pdf")
        plt.figure()

        fig5 = sns.lineplot(x='Probability', y='UCER', data=df)
        figg5 = fig5.figure
        figg5.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_means_ucer_prob.png")
        figg5.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_means_ucer_prob.pdf")
        plt.figure()

        fig6 = sns.lineplot(x='BertScore', y='F1 CRF', data=df)
        figg6 = fig6.figure
        figg6.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_means_bert_f1.png")
        figg6.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_means_bert_f1.pdf")
        plt.figure()

        fig7 = sns.lineplot(x='Probability', y='BertScore', data=df)
        figg7 = fig7.figure
        figg7.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_means_bert_prob.png")
        figg7.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_means_bert_prob.pdf")
        plt.figure()
    @staticmethod
    def all_means_prob_bleu():
        sns.set_theme()
        str = "90"
        str2 = "90.1"
        path = f"./../experiment_results/trafo{str}/exp{str2}/all_means.json"
        df = pd.read_json(path_or_buf=path)

        prob_list = ["0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5", "0.55", "0.6",
                     "0.65", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95", "1.0"]

        df["Probability"] = prob_list

        fig1 = sns.lineplot(x='Probability', y='F1 CRF', data=df)
        figg1 = fig1.figure
        figg1.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_means_prob_f1.png")
        figg1.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_means_prob_f1.pdf")
        plt.figure()
        fig2 = sns.lineplot(x='TTR', y='F1 CRF', data=df)
        figg2 = fig2.figure
        figg2.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_means_ttr_f1.png")
        figg2.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_means_ttr_f1.pdf")
        plt.figure()
        fig3 = sns.lineplot(x='Probability', y='TTR', data=df)
        figg3 = fig3.figure
        figg3.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_means_ttr_prob.png")
        figg3.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_means_ttr_prob.pdf")
        plt.figure()
        fig4 = sns.lineplot(x='UCER', y='F1 CRF', data=df)
        figg4 = fig4.figure
        figg4.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_means_ucer_f1.png")
        figg4.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_means_ucer_f1.pdf")
        plt.figure()
        fig5 = sns.lineplot(x='Probability', y='UCER', data=df)
        figg5 = fig5.figure
        figg5.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_means_ucer_prob.png")
        figg5.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_means_ucer_prob.pdf")
        plt.figure()
        fig6 = sns.lineplot(x='BleuScore', y='F1 CRF', data=df)
        figg6 = fig6.figure
        figg6.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_means_bleu_f1.png")
        figg6.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_means_bleu_f1.pdf")
        plt.figure()
        fig7 = sns.lineplot(x='Probability', y='BleuScore', data=df)
        figg7 = fig7.figure
        figg7.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_means_bleu_prob.png")
        figg7.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_means_bleu_prob.pdf")
        plt.figure()


    @staticmethod
    def all_entities_prob():
        sns.set_theme()
        str = "90"
        str2 = "90.1"
        path = f"./../experiment_results/trafo{str}/exp{str2}/all_entities_f1.json"
        df = pd.read_json(path_or_buf=path)

        prob_list = ["0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5", "0.55", "0.6",
                     "0.65", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95", "1.0"]

        df["Probability"] = prob_list

        fig1 = sns.lineplot(x='Probability', y='Actor', data=df)
        figg1 = fig1.figure
        figg1.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_ent_prob_actor.png")
        figg1.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_ent_prob_actor.pdf")
        plt.figure()

        fig1 = sns.lineplot(x='Probability', y='Activity', data=df)
        figg1 = fig1.figure
        figg1.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_ent_prob_activity.png")
        figg1.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_ent_prob_activity.pdf")
        plt.figure()

        fig1 = sns.lineplot(x='Probability', y='Activity Data', data=df)
        figg1 = fig1.figure
        figg1.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_ent_prob_actdata.png")
        figg1.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_ent_prob_actdata.pdf")
        plt.figure()

        fig1 = sns.lineplot(x='Probability', y='Further Specification', data=df)
        figg1 = fig1.figure
        figg1.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_ent_prob_further.png")
        figg1.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_ent_prob_further.pdf")
        plt.figure()

        fig1 = sns.lineplot(x='Probability', y='XOR Gateway', data=df)
        figg1 = fig1.figure
        figg1.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_ent_prob_xor.png")
        figg1.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_ent_prob_xor.pdf")
        plt.figure()

        fig1 = sns.lineplot(x='Probability', y='Condition Specification', data=df)
        figg1 = fig1.figure
        figg1.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_ent_prob_cond.png")
        figg1.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_ent_prob_cond.pdf")
        plt.figure()

        fig1 = sns.lineplot(x='Probability', y='AND Gateway', data=df)
        figg1 = fig1.figure
        figg1.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_ent_prob_and.png")
        figg1.savefig(f"./../experiment_results/trafo{str}/exp{str2}/plots/all_ent_prob_and.pdf")
        plt.figure()

sns.set_theme()
str = "3"
str2 = "3.1"
path = f"./../experiment_results/rate101/all_means.json"
df = pd.read_json(path_or_buf=path)

prob_list = ["0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5", "0.55", "0.6",
             "0.65", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95", "1.0"]
rate = [0.0, 1.0, 2.0, 3.0, 4.0]
#df["Probability"] = prob_list
df["Aug Rate"] = rate
print(df)
fig = sns.lmplot(x='Aug Rate', y='F1 CRF', data=df, fit_reg=True)
fii = fig.figure
fii.savefig("./../experiment_results/rate101/f1_rate.pdf")
fii.savefig("./../experiment_results/rate101/f1_rate.png")
plt.show()

#Plot.all_means_prob_bleu()
#Plot.all_entities_prob()
