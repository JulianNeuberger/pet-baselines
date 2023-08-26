import copy
import numpy as np
import scipy.stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy
from scipy.stats import chi2
from scipy.optimize import curve_fit
import json


# Methods for Plotting our results

# Author: Benedikt
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


# Author: Leonie
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


# Author: Leonie
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


# Author: Benedikt
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


# Author: Benedikt
def f1_norm():
    sns.set_theme()
    str = "101"
    str2 = "3"
    path = f"./../experiment_results/rate{str}/all_means.json"
    df = pd.read_json(path_or_buf=path)
    #rate = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    df["Aug Rate"] = rate

    #fig = plt.scatter(x=df["Aug Rate"], y=df["F1 CRF"])
    #plt.show()
    #fig, ax = plt.subplots(1, 1)
    mean, var, skew, kurt = norm.stats(moments='mvsk')
   # x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
    #fig.plot(x, norm.pdf(x), 'r-', lw=5, alpha=0.6, label='norm pdf')
    #ax.plot(x, norm.pdf(x), 'r-', lw=5, alpha=0.6, label='norm pdf')
    # rv = norm()
    # ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
    # r = norm.rvs(size=1000)
    #plt.hist(df["F1 CRF"], density=True, bins='auto', histtype='stepfilled', alpha=0.2)
    #ax.set_xlim([x[0], x[-1]])
    #ax.legend(loc='best', frameon=False)
    # plt.show()
    #fig = sns.lmplot(x='Aug Rate', y='F1 CRF', data=df, fit_reg=True)

    # plt.scatter(x=df["Aug Rate"], y=df["F1 CRF"])
    # plt.show()
    # std= np.std(df["F1 CRF"], ddof=1)
    # mean = np.mean(df["F1 CRF"])
    # domainx = np.linspace(np.min(df["F1 CRF"]), np.max(df["F1 CRF"]))
    #domainy = np.linspace(np.min(df["F1 CRF"]), np.max(df["F1 CRF"]))
    #domain = np.linspace()
    # plt.plot(domainx, norm.pdf(domainx, mean, std))
    # df2 = pd.DataFrame()
    #
    # plt.hist(df["F1 CRF"], edgecolor="black", alpha = 0.5, density=True)
    # plt.title("normal fit")
    # plt.xlabel("Aug Faktor")
    # plt.ylabel("F1")
    # plt.legend()
    # plt.show()
    std = np.std(df["F1 CRF"], ddof=1)
    mean = np.mean(df["F1 CRF"])
    print(mean)
    print(std)
    x = np.linspace(0, 10, 5000) # an x achse verschieben und wie viele unterteilungen
    fig, ax = plt.subplots()
    ax.plot(x, norm.pdf(x, 3, mean))
    plt.scatter(x=df["Aug Rate"], y=df["F1 CRF"])
    plt.show()
    figg = fig.figure
    figg.savefig("./../experiment_results/rate101/f1_norm.png")
    figg.savefig("./../experiment_results/rate101/f1_norm.pdf")

# Author: Benedikt
def f1_chi2():
    sns.set_theme()
    str = "101"
    path = f"./../experiment_results/rate{str}/all_means.json"
    df = pd.read_json(path_or_buf=path)
    min = np.min(df["F1 CRF"])
    df5 = copy.deepcopy(df)
    for i in range(len(df["F1 CRF"])):
        df5["F1 CRF"][i] = df5["F1 CRF"][i] - min
    rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    df5["Aug Rate"] = rate
    std = np.std(df5["F1 CRF"], ddof=1)
    mean = np.mean(df5["F1 CRF"])
    print(mean)
    print(std)
    x = np.linspace(0, 10, 5000)  # an x achse verschieben und wie viele unterteilungen
    fig, ax = plt.subplots()
    ax.plot(x, chi2.pdf(x, 5))
    plt.scatter(x=df5["Aug Rate"], y=df5["F1 CRF"])
    plt.show()
    figg = fig.figure
    figg.savefig("./../experiment_results/rate101/f1_chi2.png")
    figg.savefig("./../experiment_results/rate101/f1_chi2.pdf")

# Author: Benedikt
def calc_pearson():
    str = "101"
    str2 = "3"
    path = f"./../experiment_results/rate{str}/all_means.json"
    df = pd.read_json(path_or_buf=path)
    # rate = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    df2 = copy.deepcopy(df)
    rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]

    rateindex = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    df2.index = rateindex
    df2["Aug Rate"] = rate

    df2 = df2.drop(labels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], axis=0)
    print(df2)
    pear = scipy.stats.pearsonr(x=df2["Aug Rate"], y=df2["F1 CRF"])
    print(pear)
    df_pear = pd.DataFrame()
    df_pear["Pear"] = [pear[0], pear[1]]
    df_pear.to_json(path_or_buf=f"./../experiment_results/rate{str}/Pearson.json", indent=4)

# Author: Leonie
def plot_rate_f1():
    sns.set_theme()
    str = "101"
    str2 = "3"
    path = f"./../experiment_results/rate{str}/all_means.json"
    df = pd.read_json(path_or_buf=path)
    # rate = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    df["Aug Rate"] = rate
    fig = sns.lineplot(x=df["Aug Rate"], y=df["F1 CRF"], data=df)
    plt.show()
    figg = fig.figure
    figg.savefig("./../experiment_results/rate101/f1_rate_lineplot.png")
    figg.savefig("./../experiment_results/rate101/f1_rate_lineplot.pdf")
    plt.figure()
    fig2 = plt.scatter(x=df["Aug Rate"], y=df["F1 CRF"])
    plt.show()
    figg2 = fig2.figure
    figg2.savefig("./../experiment_results/rate101/f1_rate_scatter.png")
    figg2.savefig("./../experiment_results/rate101/f1_rate_scatter.pdf")

# Author: Benedikt
def plot_train_as_test():
    sns.set_theme()
    str = "101"
    path = f"./../experiment_results/rate5/test_all_means.json"
    path2 = f"./../experiment_results/rate82/test_all_means.json"
    path3 = f"./../experiment_results/rate100/test_all_means.json"
    path4 = f"./../experiment_results/rate101/test_all_means.json"


    df = pd.read_json(path_or_buf=path)
    df2 = pd.read_json(path_or_buf=path2)
    df3 = pd.read_json(path_or_buf=path3)
    df5 = pd.read_json(path_or_buf=path4)

    rate = [0.0, 1.0, 2.0, 3.0, 4.0]
    df4 = pd.DataFrame({"Trafo 5": df["F1 CRF"], "Trafo 82": df2["F1 CRF"], "Trafo 100": df3["F1 CRF"],
                        "Trafo 101": df5["F1 CRF"]})
    df4.index = rate

    fig = sns.lineplot(data=df4)
    fig.set(xlabel="Augmentierungsrate", ylabel="F1 Score")

    fig.set_title("Augmented Test Data")
    figg = fig.figure
    figg.savefig("./../experiment_results/Plots/train_as_testBenedikt.pdf")
    figg.savefig("./../experiment_results/Plots/train_as_testBenedikt.png")
    figg.savefig("./../experiment_results/Plots/train_as_testBenedikt.svg")
    plt.show()

# Author: Benedikt
def fn(x, c, s, df):
  return s * chi2.pdf(x, df) + c

# Author: Benedikt
def plot_with_diff_rates():
    sns.set_theme()
    str = "101"
    path05 = f"./../experiment_results/rate{str}/all_means.json"
    path025 = f"./../experiment_results/rate{str}/prob025/all_means.json"
    path075 = f"./../experiment_results/rate{str}/prob075/all_means.json"
    path10 = f"./../experiment_results/rate{str}/prob1/all_means.json"
    #path58 = f"./../experiment_results/trafo58/exp58.1/all_means.json"
    df = pd.read_json(path_or_buf=path05)
    df2 = pd.read_json(path_or_buf=path025)
    df3 = pd.read_json(path_or_buf=path075)
    df5 = pd.read_json(path_or_buf=path10)
    #df58 = pd.read_json(path_or_buf=path58)
    rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    rate2 = [0.0, 0.3, 0.6, 0.9, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    rate3 = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
                         0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    #rate2 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    #df43 = pd.DataFrame({"F1 Score": df3["F1 CRF"], "ETTR": df3["TTR"], "CETTR": df3["UCER"],
            #            "Bert Score" : df3["BertScore"]})
    df4 = pd.DataFrame({"p = 0.25": df2["F1 CRF"],"p = 0.5": df["F1 CRF"],"p = 0.75": df3["F1 CRF"],"p = 1.0": df5["F1 CRF"]})
    #df4 = pd.DataFrame(
    #    { "p = 0.5": df["F1 CRF"], "p = 0.75": df3["F1 CRF"]})
    df4.index = rate

    #min = np.min([np.min(df4["p = 0.5"]), np.min(df4["p = 0.25"]), np.min(df4["p = 0.75"]), np.min(df4["p = 1"])])
    #print(df4)
    # for index, row in df4.iterrows():
    #
    #     x = df4["Prob = 0.5"][index]
    #     df4["Prob = 0.5"][index] = x -min
    # for index, row in df4.iterrows():
    #     x = df4["Prob = 0.25"][index]
    #     df4["Prob = 0.25"][index] = x - min
    # for index, row in df4.iterrows():
    #     x = df4["Prob = 0.75"][index]
    #     df4["Prob = 0.75"][index] = x - min
    # for index, row in df4.iterrows():
    #     x = df4["Prob = 1"][index]
    #     df4["Prob = 1"][index] = x - min

    x = np.linspace(0, 10, 5000)
    #not_rate = [0.10, 0.20, 0.40, 0.50,  0.70, 0.80, 1.00]
    #not_rate = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.25, 1.50, 1.75]
    #df4 = df4.drop(index=not_rate)

    palette = sns.color_palette(["#FFAEAE", "#92A8FF", "#78D88B", "#E8D381" ],n_colors=4, desat=1)
    fig = sns.lineplot(data=df4, palette=palette)
    fig.set(xlabel="Augmentierungsrate")
    fig.set(ylabel="F1 Score")

    # popt, pcov = curve_fit(f=fn, xdata=rate2, ydata=df4["p = 0.25"])
    # popt2, pcov2 = curve_fit(f=fn, xdata=rate2, ydata=df4["p = 0.5"])
    # popt3, pcov3 = curve_fit(f=fn, xdata=rate2, ydata=df4["p = 0.75"])
    # popt4, pcov4 = curve_fit(f=fn, xdata=rate2, ydata=df4["p = 1"])



    # print(np.linalg.cond(pcov))
    # print(np.linalg.cond(pcov2))
    # print(np.linalg.cond(pcov3))
    # print(np.linalg.cond(pcov4))


    # fit = fn(rate2, *popt)
    # fit2 = fn(rate2, *popt2)
    # fit3 = fn(rate2, *popt3)
    # fit4 = fn(rate2, *popt4)

    # #goodness1 = scipy.stats.goodness_of_fit(fit, df4["p = 0.25"])
    # #print(goodness1)
    # max_x = fit.argmax(axis=0)
    # max_x2 = fit2.argmax(axis=0)
    # max_x3 = fit3.argmax(axis=0)
    # max_x4 = fit4.argmax(axis=0)
    #
    # best_aug = rate2[max_x]
    # best_aug2 = rate2[max_x2]
    # best_aug3 = rate2[max_x3]
    # best_aug4 = rate2[max_x4]

    # df_best_rate = pd.DataFrame()
    # series_rate = pd.Series([best_aug, df4["p = 0.25"][best_aug]])
    # df_best_rate = df_best_rate.append(series_rate,ignore_index=True)
    # series_rate2 = pd.Series([best_aug, df4["p = 0.5"][best_aug2]])
    # df_best_rate = df_best_rate.append(series_rate2,ignore_index=True)
    # series_rate3 = pd.Series([best_aug, df4["p = 0.75"][best_aug3]])
    # df_best_rate = df_best_rate.append(series_rate3,ignore_index=True)
    # series_rate4 = pd.Series([best_aug, df4["p = 1"][best_aug4]])
    # df_best_rate = df_best_rate.append(series_rate4, ignore_index=True)

    # df_best_rate = df_best_rate.set_axis(["Best Aug Rate", "F1 Score"], axis=1)
    # df_best_rate.index = ["p = 0.25", "p = 0.5", "p = 0.75", "p = 1"]
    # print(df_best_rate)
    # df_best_rate.to_json(path_or_buf=f"./../experiment_results/trafo{str}/best_aug_rate.json", indent=4)
    #
    #
    #
    # fig.plot(rate2, fit, color="#FF1B1B", label="Chi2 p = 0.25")
    # fig.plot(rate2, fit2, color="#1846FD", label="Chi2 p = 0.5")
    # fig.plot(rate2, fit3, color="#0EBC30", label="Chi2 p = 0.75")
    # fig.plot(rate2, fit4, color="#DFB81A", label="Chi2 p = 1")

    fig.legend()
    #fig.plot(rate, fit, color="black")

    fig.set_title(f"Transformation 101")

    #fig.plot(x, chi2.pdf(x, 5), color="black")

    figg = fig.figure
    figg.savefig(f"./../experiment_results/trafo101/plots/101.pdf")
    figg.savefig(f"./../experiment_results/trafo101/plots/101.png")
    figg.savefig(f"./../experiment_results/trafo101/plots/101.svg")
    plt.show()


# Author: Leonie
def get_df_with_all_ttr_means_per_prob_rate(trafo_nr=101):
    #trafo_nr = "5"
    df_all_ttr_means = pd.DataFrame()
    rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]

    list1 = []
    for i in rate:
        path = f"./../experiment_results/rate{trafo_nr}/prob025/ttr_mean_un_{i}.json"
        df = pd.read_json(path_or_buf=path)
        list1.append(df["All"][0])
    new_series1 = pd.Series(list1)
    df_all_ttr_means = df_all_ttr_means.append(new_series1, ignore_index=True )

    list4= []
    for i in rate:
        path = f"./../experiment_results/rate{trafo_nr}/ttr_mean_un_{i}.json"
        df = pd.read_json(path_or_buf=path)
        list4.append(df["All"][0])
    new_series4 = pd.Series(list4)
    df_all_ttr_means = df_all_ttr_means.append(new_series4, ignore_index=True)

    list2 = []
    for i in rate:
        path = f"./../experiment_results/rate{trafo_nr}/prob075/ttr_mean_un_{i}.json"
        df = pd.read_json(path_or_buf=path)
        list2.append(df["All"][0])
    new_series2 = pd.Series(list2)
    df_all_ttr_means = df_all_ttr_means.append(new_series2, ignore_index=True)

    list3 = []
    for i in rate:
        path = f"./../experiment_results/rate{trafo_nr}/prob1/ttr_mean_un_{i}.json"
        df = pd.read_json(path_or_buf=path)
        list3.append(df["All"][0])
    new_series3 = pd.Series(list3)
    df_all_ttr_means = df_all_ttr_means.append(new_series3, ignore_index=True)

    df_all_ttr_means = df_all_ttr_means.set_axis(rate, axis=1)
    df_all_ttr_means.index = [0.25, 0.5, 0.75, 1.0]
    print(df_all_ttr_means)
    df_all_ttr_means.to_json(path_or_buf=f"./../experiment_results/rate{trafo_nr}/aaa_ttr_means_un_per_prob_and_rate.json", indent=4)


    path1 = f"./../experiment_results/rate{trafo_nr}/prob025/all_means.json"
    path2 = f"./../experiment_results/rate{trafo_nr}/all_means.json"
    path3 = f"./../experiment_results/rate{trafo_nr}/prob075/all_means.json"
    path4 = f"./../experiment_results/rate{trafo_nr}/prob1/all_means.json"

    df6 = pd.read_json(path_or_buf=path1)
    df7 = pd.read_json(path_or_buf=path2)
    df8 = pd.read_json(path_or_buf=path3)
    df9 = pd.read_json(path_or_buf=path4)
    df6.index = rate
    df7.index = rate
    df8.index = rate
    df9.index = rate

    df_new_all_means_aug_ttr = pd.DataFrame()
    new_ser1 = pd.Series(df6["TTR"])
    df_new_all_means_aug_ttr = df_new_all_means_aug_ttr.append(new_ser1, ignore_index=True)
    new_ser2 = pd.Series(df7["TTR"])
    df_new_all_means_aug_ttr = df_new_all_means_aug_ttr.append(new_ser2, ignore_index=True)
    new_ser3 = pd.Series(df8["TTR"])
    df_new_all_means_aug_ttr = df_new_all_means_aug_ttr.append(new_ser3, ignore_index=True)
    new_ser4 = pd.Series(df9["TTR"])
    df_new_all_means_aug_ttr = df_new_all_means_aug_ttr.append(new_ser4, ignore_index=True)

    df_new_all_means_aug_ttr.index = [0.25, 0.5, 0.75, 1.0]
    df_new_all_means_aug_ttr.to_json(path_or_buf=f"./../experiment_results/rate{trafo_nr}/aaa_ttr_means_per_prob_and_rate.json", indent=4)

# Author: Leonie
def get_df_with_all_ucer_means_per_prob_rate(trafo_nr):
    # trafo_nr = "5"
    df_all_ttr_means = pd.DataFrame()
    rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]

    list1 = []
    for i in rate:
        path = f"./../experiment_results/rate{trafo_nr}/prob025/ucer_mean_un_{i}.json"
        df = pd.read_json(path_or_buf=path)
        list1.append(df["All"][0])
    new_series1 = pd.Series(list1)
    df_all_ttr_means = df_all_ttr_means.append(new_series1, ignore_index=True)

    list4 = []
    for i in rate:
        path = f"./../experiment_results/rate{trafo_nr}/ucer_mean_un_{i}.json"
        df = pd.read_json(path_or_buf=path)
        list4.append(df["All"][0])
    new_series4 = pd.Series(list4)
    df_all_ttr_means = df_all_ttr_means.append(new_series4, ignore_index=True)

    list2 = []
    for i in rate:
        path = f"./../experiment_results/rate{trafo_nr}/prob075/ucer_mean_un_{i}.json"
        df = pd.read_json(path_or_buf=path)
        list2.append(df["All"][0])
    new_series2 = pd.Series(list2)
    df_all_ttr_means = df_all_ttr_means.append(new_series2, ignore_index=True)

    list3 = []
    for i in rate:
        path = f"./../experiment_results/rate{trafo_nr}/prob1/ucer_mean_un_{i}.json"
        df = pd.read_json(path_or_buf=path)
        list3.append(df["All"][0])
    new_series3 = pd.Series(list3)
    df_all_ttr_means = df_all_ttr_means.append(new_series3, ignore_index=True)

    df_all_ttr_means = df_all_ttr_means.set_axis(rate, axis = 1)
    df_all_ttr_means.index = [0.25, 0.5, 0.75, 1.0]

    df_all_ttr_means.to_json(
        path_or_buf=f"./../experiment_results/rate{trafo_nr}/aaa_ucer_means_un_per_prob_and_rate.json", indent=4)

    path1 = f"./../experiment_results/rate{trafo_nr}/prob025/all_means.json"
    path2 = f"./../experiment_results/rate{trafo_nr}/all_means.json"
    path3 = f"./../experiment_results/rate{trafo_nr}/prob075/all_means.json"
    path4 = f"./../experiment_results/rate{trafo_nr}/prob1/all_means.json"

    df6 = pd.read_json(path_or_buf=path1)
    df7 = pd.read_json(path_or_buf=path2)
    df8 = pd.read_json(path_or_buf=path3)
    df9 = pd.read_json(path_or_buf=path4)
    df6.index = rate
    df7.index = rate
    df8.index = rate
    df9.index = rate


    df_new_all_means_aug_ttr = pd.DataFrame()
    new_ser1 = pd.Series(df6["UCER"])
    df_new_all_means_aug_ttr = df_new_all_means_aug_ttr.append(new_ser1, ignore_index=True)
    new_ser2 = pd.Series(df7["UCER"])
    df_new_all_means_aug_ttr = df_new_all_means_aug_ttr.append(new_ser2, ignore_index=True)
    new_ser3 = pd.Series(df8["UCER"])
    df_new_all_means_aug_ttr = df_new_all_means_aug_ttr.append(new_ser3, ignore_index=True)
    new_ser4 = pd.Series(df9["UCER"])
    df_new_all_means_aug_ttr = df_new_all_means_aug_ttr.append(new_ser4, ignore_index=True)

    df_new_all_means_aug_ttr.index = [0.25, 0.5, 0.75, 1.0]
    df_new_all_means_aug_ttr.to_json(
        path_or_buf=f"./../experiment_results/rate{trafo_nr}/aaa_ucer_means_per_prob_and_rate.json", indent=4)

# Author: Leonie
def get_df_with_all_bleu_means_per_prob_rate(trafo_nr):
    rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]

    path1 = f"./../experiment_results/rate{trafo_nr}/prob025/all_means.json"
    path2 = f"./../experiment_results/rate{trafo_nr}/all_means.json"
    path3 = f"./../experiment_results/rate{trafo_nr}/prob075/all_means.json"
    path4 = f"./../experiment_results/rate{trafo_nr}/prob1/all_means.json"

    df6 = pd.read_json(path_or_buf=path1)
    df7 = pd.read_json(path_or_buf=path2)
    df8 = pd.read_json(path_or_buf=path3)
    df9 = pd.read_json(path_or_buf=path4)
    df6.index = rate
    df7.index = rate
    df8.index = rate
    df9.index = rate

    df_new_all_means_aug_ttr = pd.DataFrame()
    new_ser1 = pd.Series(df6["BleuScore"])
    df_new_all_means_aug_ttr = df_new_all_means_aug_ttr.append(new_ser1, ignore_index=True)
    new_ser2 = pd.Series(df7["BleuScore"])
    df_new_all_means_aug_ttr = df_new_all_means_aug_ttr.append(new_ser2, ignore_index=True)
    new_ser3 = pd.Series(df8["BleuScore"])
    df_new_all_means_aug_ttr = df_new_all_means_aug_ttr.append(new_ser3, ignore_index=True)
    new_ser4 = pd.Series(df9["BleuScore"])
    df_new_all_means_aug_ttr = df_new_all_means_aug_ttr.append(new_ser4, ignore_index=True)

    df_new_all_means_aug_ttr = df_new_all_means_aug_ttr.set_axis(rate, axis=1)
    df_new_all_means_aug_ttr.index = [0.25, 0.5, 0.75, 1.0]
    df_new_all_means_aug_ttr.to_json(
        path_or_buf=f"./../experiment_results/rate{trafo_nr}/aaa_bleu_means_per_prob_and_rate.json", indent=4)

# Author: Benedikt
def get_df_with_all_bert_means_per_prob_rate(trafo_nr):
    rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]

    path1 = f"./../experiment_results/rate{trafo_nr}/prob025/all_means.json"
    path2 = f"./../experiment_results/rate{trafo_nr}/all_means.json"
    path3 = f"./../experiment_results/rate{trafo_nr}/prob075/all_means.json"
    path4 = f"./../experiment_results/rate{trafo_nr}/prob1/all_means.json"

    df6 = pd.read_json(path_or_buf=path1)
    df7 = pd.read_json(path_or_buf=path2)
    df8 = pd.read_json(path_or_buf=path3)
    df9 = pd.read_json(path_or_buf=path4)
    df6.index = rate
    df7.index = rate
    df8.index = rate
    df9.index = rate

    df_new_all_means_aug_ttr = pd.DataFrame()
    new_ser1 = pd.Series(df6["BertScore"])
    df_new_all_means_aug_ttr = df_new_all_means_aug_ttr.append(new_ser1, ignore_index=True)
    new_ser2 = pd.Series(df7["BertScore"])
    df_new_all_means_aug_ttr = df_new_all_means_aug_ttr.append(new_ser2, ignore_index=True)
    new_ser3 = pd.Series(df8["BertScore"])
    df_new_all_means_aug_ttr = df_new_all_means_aug_ttr.append(new_ser3, ignore_index=True)
    new_ser4 = pd.Series(df9["BertScore"])
    df_new_all_means_aug_ttr = df_new_all_means_aug_ttr.append(new_ser4, ignore_index=True)

    df_new_all_means_aug_ttr = df_new_all_means_aug_ttr.set_axis(rate, axis=1)
    df_new_all_means_aug_ttr.index = [0.25, 0.5, 0.75, 1.0]
    df_new_all_means_aug_ttr.to_json(
        path_or_buf=f"./../experiment_results/rate{trafo_nr}/aaa_bert_means_per_prob_and_rate.json", indent=4)

# Author: Leonie
def plot_ttr_means_per_prob_and_rate_ratetottr(trafo_nr=101):
    sns.set_theme()
    #trafo_nr = "39"
    rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]

    path = f"./../experiment_results/rate{trafo_nr}/aaa_ttr_means_un_per_prob_and_rate.json"
    df1 = pd.read_json(path_or_buf=path)
    path2 = f"./../experiment_results/rate{trafo_nr}/aaa_ttr_means_per_prob_and_rate.json"
    df2 = pd.read_json(path_or_buf=path2)

    df2 = df2.subtract(df1)
    df2 = df2.set_axis(rate, axis=1)
    print(df2)
    df2.index = [0.25, 0.5, 0.75, 1.0]
    df2.to_json(path_or_buf=f"./../experiment_results/rate{trafo_nr}/aaa_ttr_means_subt_per_prob_and_rate.json", indent=4)

    df4 = pd.DataFrame({"Prob 0.25": df2.iloc[0], "Prob 0.5": df2.iloc[1], "Prob 0.75": df2.iloc[2], "Prob 1": df2.iloc[3]})
    df4.index = rate

    fig = sns.lineplot(data=df4)
    fig.set(xlabel="Augmentierungsrate", ylabel="ETTR difference")

    figg = fig.figure
    figg.tight_layout()
    figg.savefig(f"./../experiment_results/trafo{trafo_nr}/plots/ttr_per_prob.pdf")
    figg.savefig(f"./../experiment_results/trafo{trafo_nr}/plots/ttr_per_prob.png")
    figg.savefig(f"./../experiment_results/trafo{trafo_nr}/plots/ttr_per_prob.svg")
    plt.show()

# Author: Leonie
def plot_ucer_means_per_prob_and_rate_ratetottr(trafo_nr):
    sns.set_theme()
    # trafo_nr = "39"
    rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]

    path = f"./../experiment_results/rate{trafo_nr}/aaa_ucer_means_un_per_prob_and_rate.json"
    df1 = pd.read_json(path_or_buf=path)
    path2 = f"./../experiment_results/rate{trafo_nr}/aaa_ucer_means_per_prob_and_rate.json"
    df2 = pd.read_json(path_or_buf=path2)

    df2 = df2.subtract(df1)
    df2 = df2.set_axis(rate, axis=1)
    df2.index = [0.25, 0.5, 0.75, 1.0]
    df2.to_json(path_or_buf=f"./../experiment_results/rate{trafo_nr}/aaa_ucer_means_subt_per_prob_and_rate.json",
                indent=4)

    df4 = pd.DataFrame(
        {"Prob 0.25": df2.iloc[0], "Prob 0.5": df2.iloc[1], "Prob 0.75": df2.iloc[2], "Prob 1": df2.iloc[3]})
    df4.index = rate

    fig = sns.lineplot(data=df4)
    fig.set(xlabel="Augmentierungsrate", ylabel="CETTR difference")

    figg = fig.figure
    figg.tight_layout()
    figg.savefig(f"./../experiment_results/trafo{trafo_nr}/plots/ucer_per_prob.pdf")
    figg.savefig(f"./../experiment_results/trafo{trafo_nr}/plots/ucer_per_prob.png")
    figg.savefig(f"./../experiment_results/trafo{trafo_nr}/plots/ucer_per_prob.svg")
    plt.show()


# Author: Leonie
def plot_ttr_mean_prob_to_ttr():
    sns.set_theme()
    trafo_nr = "3"
    name = "ttr"
    name2 = "ttr"
    rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]

    path = f"./../experiment_results/rate3/aaa_{name2}_means_per_prob_and_rate.json"
    df = pd.read_json(path_or_buf=path)
    df = df.set_axis(rate, axis=1)
    #df.loc[0.00] = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    print(df)
    df.index = [ 0.25, 0.5, 0.75, 1.0]
    # df = df.sort_index().reset_index(drop=True)
    # df.index = [0.00, 0.25, 0.5, 0.75, 1.0]

    path2 = f"./../experiment_results/rate39/aaa_{name2}_means_per_prob_and_rate.json"
    df2 = pd.read_json(path_or_buf=path2)
    df2 = df2.set_axis(rate, axis=1)
   # df2.loc[0.00] = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    df2.index = [0.25, 0.5, 0.75, 1.0]
    # df2 = df2.sort_index().reset_index(drop=True)
    # df2.index = [0.00, 0.25, 0.5, 0.75, 1.0]

    path3 = f"./../experiment_results/rate40/aaa_{name2}_means_per_prob_and_rate.json"
    df3 = pd.read_json(path_or_buf=path3)
    df3 = df3.set_axis(rate, axis=1)
    #df3.loc[0.00] = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    df3.index = [0.25, 0.5, 0.75, 1.0]
    # df3 = df3.sort_index().reset_index(drop=True)
    # df3.index = [0.00, 0.25, 0.5, 0.75, 1.0]

    path4 = f"./../experiment_results/rate86/aaa_{name2}_means_per_prob_and_rate.json"
    df4 = pd.read_json(path_or_buf=path4)
    df4 = df4.set_axis(rate, axis=1)
    #df4.loc[0.00] = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    df4.index = [0.25, 0.5, 0.75, 1.0]
    # df4 = df4.sort_index().reset_index(drop=True)
    # df4.index = [0.00, 0.25, 0.5, 0.75, 1.0]

    path5 = f"./../experiment_results/rate90/aaa_{name2}_means_per_prob_and_rate.json"
    df5 = pd.read_json(path_or_buf=path5)
    df5 = df5.set_axis(rate, axis=1)
    #df5.loc[0.00] = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    df5.index = [0.25, 0.5, 0.75, 1.0]
    # df5 = df5.sort_index().reset_index(drop=True)
    # df5.index = [0.00, 0.25, 0.5, 0.75, 1.0]

    path6 = f"./../experiment_results/rate103/aaa_{name2}_means_per_prob_and_rate.json"
    df6 = pd.read_json(path_or_buf=path6)
    df6 = df6.set_axis(rate, axis=1)
    #df6.loc[0.00] = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    df6.index = [0.25, 0.5, 0.75, 1.0]
    # df6 = df6.sort_index().reset_index(drop=True)
    # df6.index = [0.00, 0.25, 0.5, 0.75, 1.0]

    path7 = f"./../experiment_results/rate5/aaa_{name}_means_per_prob_and_rate.json"
    df7 = pd.read_json(path_or_buf=path7)
    df7 = df7.set_axis(rate, axis=1)
    #df7.loc[0.00] = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    df7.index = [0.25, 0.5, 0.75, 1.0]
    # df7 = df7.sort_index().reset_index(drop=True)
    # df7.index = [0.00, 0.25, 0.5, 0.75, 1.0]

    path8 = f"./../experiment_results/rate82/aaa_{name}_means_per_prob_and_rate.json"
    df8 = pd.read_json(path_or_buf=path8)
    df8 = df8.set_axis(rate, axis=1)
    #df8.loc[0.00] = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    df8.index = [0.25, 0.5, 0.75, 1.0]
    # df8 = df8.sort_index().reset_index(drop=True)
    # df8.index = [0.00, 0.25, 0.5, 0.75, 1.0]

    path9 = f"./../experiment_results/rate100/aaa_{name}_means_per_prob_and_rate.json"
    df9 = pd.read_json(path_or_buf=path9)
    df9 = df9.set_axis(rate, axis=1)
    #df9.loc[0.00] = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    df9.index = [0.25, 0.5, 0.75, 1.0]
    # df9 = df9.sort_index().reset_index(drop=True)
    # df9.index = [0.00, 0.25, 0.5, 0.75, 1.0]

    path10 = f"./../experiment_results/rate101/aaa_{name}_means_per_prob_and_rate.json"
    df10 = pd.read_json(path_or_buf=path10)
    df10 = df10.set_axis(rate, axis=1)
    #df10.loc[0.00] = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    df10.index = [0.25, 0.5, 0.75, 1.0]
    # df10 = df10.sort_index().reset_index(drop=True)
    # df10.index = [0.00, 0.25, 0.5, 0.75, 1.0]

    path11 = f"./../experiment_results/trafo58/exp58.1/all_means.json"
    df11 = pd.read_json(path_or_buf=path11)
    dff = pd.Series([df11["ETTR"][4],df11["ETTR"][9],df11["ETTR"][14],df11["ETTR"][19]  ])



    dff.index = [0.25, 0.5, 0.75, 1.0]

    print(df)
    #full_df = pd.DataFrame({"Trafo 3": df[3.0], "Trafo 39": df2[3.0],"Trafo 40": df3[3.0],"Trafo 86": df4[3.0],
    #                       "Trafo 90": df5[3.0],"Trafo 103": df6[3.0]})
    full_df = pd.DataFrame({"Trafo 5": df7[3.0], "Trafo 82": df8[3.0], "Trafo 100": df9[3.0], "Trafo 101": df10[3.0], "Trafo 58": dff})
    fig = sns.lineplot(data=full_df)
    fig.set(xlabel="Ersetzungswahrscheinlichkeit", ylabel=f"ETTR")
    figg = fig.figure
    figg.tight_layout()
    figg.savefig(f"./../experiment_results/Plots/Benedikt/{name}_per_prob_new.pdf")
    figg.savefig(f"./../experiment_results/Plots/Benedikt/{name}_per_prob_new.png")
    figg.savefig(f"./../experiment_results/Plots/Benedikt/{name}_per_prob_new.svg")
    plt.show()


# Author: Benedikt
def plot_bert_means_per_prob_and_rate_ratetottr():
    sns.set_theme()
    trafo_nr = "103"
    rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]

    path = f"./../experiment_results/rate{trafo_nr}/aaa_bleu_means_per_prob_and_rate.json"
    df1 = pd.read_json(path_or_buf=path)
    df4 = pd.DataFrame(
        {"Prob 0.25": df1.iloc[0], "Prob 0.5": df1.iloc[1], "Prob 0.75": df1.iloc[2], "Prob 1": df1.iloc[3]})
    df4.index = rate
    print(df4)
    fig = sns.lineplot(data=df4)
    fig.set(xlabel="Augmentierungsrate", ylabel="Bleu Score")

    # figg = fig.figure
    # figg.tight_layout()
    # figg.savefig(f"./../experiment_results/trafo{trafo_nr}/plots/bleu_per_prob.pdf")
    # figg.savefig(f"./../experiment_results/trafo{trafo_nr}/plots/bleu_per_prob.png")
    # figg.savefig(f"./../experiment_results/trafo{trafo_nr}/plots/bleu_per_prob.svg")
    plt.show()

# Author: Benedikt
def scatter():
    sns.set_theme()
    bert_list = []
    bleu_list = []
    f1_list = []
    f12_list = []
    ucer_list = []
    ttr_list = []
    ttr_list2 = []
    ucer_list2 = []

    trafo_nr = "101"
    path = f"./../experiment_results/rate{trafo_nr}/prob025/all_means.json"
    path2 = f"./../experiment_results/rate{trafo_nr}/all_means.json"
    path3 = f"./../experiment_results/rate{trafo_nr}/prob075/all_means.json"
    path4 = f"./../experiment_results/rate{trafo_nr}/prob1/all_means.json"
    df1 = pd.read_json(path_or_buf=path)
    df2 = pd.read_json(path_or_buf=path2)
    df3 = pd.read_json(path_or_buf=path3)
    df4 = pd.read_json(path_or_buf=path4)
    bert_list.extend(df1["BertScore"].tolist())
    f1_list.extend(df1["F1 CRF"].tolist())
    ttr_list.extend(df1["TTR"].tolist())
    ucer_list.extend(df1["UCER"].tolist())

    bert_list.extend(df2["BertScore"].tolist())
    f1_list.extend(df2["F1 CRF"].tolist())
    ttr_list.extend(df2["TTR"].tolist())
    ucer_list.extend(df2["UCER"].tolist())

    bert_list.extend(df3["BertScore"].tolist())
    f1_list.extend(df3["F1 CRF"].tolist())
    ttr_list.extend(df3["TTR"].tolist())
    ucer_list.extend(df3["UCER"].tolist())

    bert_list.extend(df4["BertScore"].tolist())
    f1_list.extend(df4["F1 CRF"].tolist())
    ttr_list.extend(df4["TTR"].tolist())
    ucer_list.extend(df4["UCER"].tolist())

    trafo_nr = "82"
    path = f"./../experiment_results/rate{trafo_nr}/prob025/all_means.json"
    path2 = f"./../experiment_results/rate{trafo_nr}/all_means.json"
    path3 = f"./../experiment_results/rate{trafo_nr}/prob075/all_means.json"
    path4 = f"./../experiment_results/rate{trafo_nr}/prob1/all_means.json"
    df1 = pd.read_json(path_or_buf=path)
    df2 = pd.read_json(path_or_buf=path2)
    df3 = pd.read_json(path_or_buf=path3)
    df4 = pd.read_json(path_or_buf=path4)
    bert_list.extend(df1["BertScore"].tolist())
    f1_list.extend(df1["F1 CRF"].tolist())
    ttr_list.extend(df1["TTR"].tolist())
    ucer_list.extend(df1["UCER"].tolist())

    bert_list.extend(df2["BertScore"].tolist())
    f1_list.extend(df2["F1 CRF"].tolist())
    ttr_list.extend(df2["TTR"].tolist())
    ucer_list.extend(df2["UCER"].tolist())

    bert_list.extend(df3["BertScore"].tolist())
    f1_list.extend(df3["F1 CRF"].tolist())
    ttr_list.extend(df3["TTR"].tolist())
    ucer_list.extend(df3["UCER"].tolist())

    bert_list.extend(df4["BertScore"].tolist())
    f1_list.extend(df4["F1 CRF"].tolist())
    ttr_list.extend(df4["TTR"].tolist())
    ucer_list.extend(df4["UCER"].tolist())

    trafo_nr = "100"
    path = f"./../experiment_results/rate{trafo_nr}/prob025/all_means.json"
    path2 = f"./../experiment_results/rate{trafo_nr}/all_means.json"
    path3 = f"./../experiment_results/rate{trafo_nr}/prob075/all_means.json"
    path4 = f"./../experiment_results/rate{trafo_nr}/prob1/all_means.json"
    df1 = pd.read_json(path_or_buf=path)
    df2 = pd.read_json(path_or_buf=path2)
    df3 = pd.read_json(path_or_buf=path3)
    df4 = pd.read_json(path_or_buf=path4)
    bert_list.extend(df1["BertScore"].tolist())
    f1_list.extend(df1["F1 CRF"].tolist())
    ttr_list.extend(df1["TTR"].tolist())
    ucer_list.extend(df1["UCER"].tolist())

    bert_list.extend(df2["BertScore"].tolist())
    f1_list.extend(df2["F1 CRF"].tolist())
    ttr_list.extend(df2["TTR"].tolist())
    ucer_list.extend(df2["UCER"].tolist())

    bert_list.extend(df3["BertScore"].tolist())
    f1_list.extend(df3["F1 CRF"].tolist())
    ttr_list.extend(df3["TTR"].tolist())
    ucer_list.extend(df3["UCER"].tolist())

    bert_list.extend(df4["BertScore"].tolist())
    f1_list.extend(df4["F1 CRF"].tolist())
    ttr_list.extend(df4["TTR"].tolist())
    ucer_list.extend(df4["UCER"].tolist())

    trafo_nr = "5"
    path = f"./../experiment_results/rate{trafo_nr}/prob025/all_means.json"
    path2 = f"./../experiment_results/rate{trafo_nr}/all_means.json"
    path3 = f"./../experiment_results/rate{trafo_nr}/prob075/all_means.json"
    path4 = f"./../experiment_results/rate{trafo_nr}/prob1/all_means.json"
    df1 = pd.read_json(path_or_buf=path)
    df2 = pd.read_json(path_or_buf=path2)
    df3 = pd.read_json(path_or_buf=path3)
    df4 = pd.read_json(path_or_buf=path4)
    bert_list.extend(df1["BertScore"].tolist())
    f1_list.extend(df1["F1 CRF"].tolist())
    ttr_list.extend(df1["TTR"].tolist())
    ucer_list.extend(df1["UCER"].tolist())

    bert_list.extend(df2["BertScore"].tolist())
    f1_list.extend(df2["F1 CRF"].tolist())
    ttr_list.extend(df2["TTR"].tolist())
    ucer_list.extend(df2["UCER"].tolist())

    bert_list.extend(df3["BertScore"].tolist())
    f1_list.extend(df3["F1 CRF"].tolist())
    ttr_list.extend(df3["TTR"].tolist())
    ucer_list.extend(df3["UCER"].tolist())

    bert_list.extend(df4["BertScore"].tolist())
    f1_list.extend(df4["F1 CRF"].tolist())
    ttr_list.extend(df4["TTR"].tolist())
    ucer_list.extend(df4["UCER"].tolist())




    trafo_nr = "3"
    path = f"./../experiment_results/rate{trafo_nr}/prob025/all_means.json"
    path2 = f"./../experiment_results/rate{trafo_nr}/all_means.json"
    path3 = f"./../experiment_results/rate{trafo_nr}/prob075/all_means.json"
    path4 = f"./../experiment_results/rate{trafo_nr}/prob1/all_means.json"
    df1 = pd.read_json(path_or_buf=path)
    df2 = pd.read_json(path_or_buf=path2)
    df3 = pd.read_json(path_or_buf=path3)
    df4 = pd.read_json(path_or_buf=path4)
    bleu_list.extend(df1["BleuScore"].tolist())
    f12_list.extend(df1["F1 CRF"].tolist())
    ttr_list2.extend(df1["TTR"].tolist())
    ucer_list2.extend(df1["UCER"].tolist())

    bleu_list.extend(df2["BleuScore"].tolist())
    f12_list.extend(df2["F1 CRF"].tolist())
    ttr_list2.extend(df2["TTR"].tolist())
    ucer_list2.extend(df2["UCER"].tolist())

    bleu_list.extend(df3["BleuScore"].tolist())
    f12_list.extend(df3["F1 CRF"].tolist())
    ttr_list2.extend(df3["TTR"].tolist())
    ucer_list2.extend(df3["UCER"].tolist())

    bleu_list.extend(df4["BleuScore"].tolist())
    f12_list.extend(df4["F1 CRF"].tolist())
    ttr_list2.extend(df4["TTR"].tolist())
    ucer_list2.extend(df4["UCER"].tolist())




    trafo_nr = "39"
    path = f"./../experiment_results/rate{trafo_nr}/prob025/all_means.json"
    path2 = f"./../experiment_results/rate{trafo_nr}/all_means.json"
    path3 = f"./../experiment_results/rate{trafo_nr}/prob075/all_means.json"
    path4 = f"./../experiment_results/rate{trafo_nr}/prob1/all_means.json"
    df1 = pd.read_json(path_or_buf=path)
    df2 = pd.read_json(path_or_buf=path2)
    df3 = pd.read_json(path_or_buf=path3)
    df4 = pd.read_json(path_or_buf=path4)
    bleu_list.extend(df1["BleuScore"].tolist())
    f12_list.extend(df1["F1 CRF"].tolist())
    ttr_list2.extend(df1["TTR"].tolist())
    ucer_list2.extend(df1["UCER"].tolist())

    bleu_list.extend(df2["BleuScore"].tolist())
    f12_list.extend(df2["F1 CRF"].tolist())
    ttr_list2.extend(df2["TTR"].tolist())
    ucer_list2.extend(df2["UCER"].tolist())

    bleu_list.extend(df3["BleuScore"].tolist())
    f12_list.extend(df3["F1 CRF"].tolist())
    ttr_list2.extend(df3["TTR"].tolist())
    ucer_list2.extend(df3["UCER"].tolist())

    bleu_list.extend(df4["BleuScore"].tolist())
    f12_list.extend(df4["F1 CRF"].tolist())
    ttr_list2.extend(df4["TTR"].tolist())
    ucer_list2.extend(df4["UCER"].tolist())




    trafo_nr = "40"
    path = f"./../experiment_results/rate{trafo_nr}/prob025/all_means.json"
    path2 = f"./../experiment_results/rate{trafo_nr}/all_means.json"
    path3 = f"./../experiment_results/rate{trafo_nr}/prob075/all_means.json"
    path4 = f"./../experiment_results/rate{trafo_nr}/prob1/all_means.json"
    df1 = pd.read_json(path_or_buf=path)
    df2 = pd.read_json(path_or_buf=path2)
    df3 = pd.read_json(path_or_buf=path3)
    df4 = pd.read_json(path_or_buf=path4)
    bleu_list.extend(df1["BleuScore"].tolist())
    f12_list.extend(df1["F1 CRF"].tolist())
    ttr_list2.extend(df1["TTR"].tolist())
    ucer_list2.extend(df1["UCER"].tolist())

    bleu_list.extend(df2["BleuScore"].tolist())
    f12_list.extend(df2["F1 CRF"].tolist())
    ttr_list2.extend(df2["TTR"].tolist())
    ucer_list2.extend(df2["UCER"].tolist())

    bleu_list.extend(df3["BleuScore"].tolist())
    f12_list.extend(df3["F1 CRF"].tolist())
    ttr_list2.extend(df3["TTR"].tolist())
    ucer_list2.extend(df3["UCER"].tolist())

    bleu_list.extend(df4["BleuScore"].tolist())
    f12_list.extend(df4["F1 CRF"].tolist())
    ttr_list2.extend(df4["TTR"].tolist())
    ucer_list2.extend(df4["UCER"].tolist())




    trafo_nr = "86"
    path = f"./../experiment_results/rate{trafo_nr}/prob025/all_means.json"
    path2 = f"./../experiment_results/rate{trafo_nr}/all_means.json"
    path3 = f"./../experiment_results/rate{trafo_nr}/prob075/all_means.json"
    path4 = f"./../experiment_results/rate{trafo_nr}/prob1/all_means.json"
    df1 = pd.read_json(path_or_buf=path)
    df2 = pd.read_json(path_or_buf=path2)
    df3 = pd.read_json(path_or_buf=path3)
    df4 = pd.read_json(path_or_buf=path4)
    bleu_list.extend(df1["BleuScore"].tolist())
    f12_list.extend(df1["F1 CRF"].tolist())
    ttr_list2.extend(df1["TTR"].tolist())
    ucer_list2.extend(df1["UCER"].tolist())

    bleu_list.extend(df2["BleuScore"].tolist())
    f12_list.extend(df2["F1 CRF"].tolist())
    ttr_list2.extend(df2["TTR"].tolist())
    ucer_list2.extend(df2["UCER"].tolist())

    bleu_list.extend(df3["BleuScore"].tolist())
    f12_list.extend(df3["F1 CRF"].tolist())
    ttr_list2.extend(df3["TTR"].tolist())
    ucer_list2.extend(df3["UCER"].tolist())

    bleu_list.extend(df4["BleuScore"].tolist())
    f12_list.extend(df4["F1 CRF"].tolist())
    ttr_list2.extend(df4["TTR"].tolist())
    ucer_list2.extend(df4["UCER"].tolist())




    trafo_nr = "90"
    path = f"./../experiment_results/rate{trafo_nr}/prob025/all_means.json"
    path2 = f"./../experiment_results/rate{trafo_nr}/all_means.json"
    path3 = f"./../experiment_results/rate{trafo_nr}/prob075/all_means.json"
    path4 = f"./../experiment_results/rate{trafo_nr}/prob1/all_means.json"
    df1 = pd.read_json(path_or_buf=path)
    df2 = pd.read_json(path_or_buf=path2)
    df3 = pd.read_json(path_or_buf=path3)
    df4 = pd.read_json(path_or_buf=path4)
    bleu_list.extend(df1["BleuScore"].tolist())
    f12_list.extend(df1["F1 CRF"].tolist())
    ttr_list2.extend(df1["TTR"].tolist())
    ucer_list2.extend(df1["UCER"].tolist())

    bleu_list.extend(df2["BleuScore"].tolist())
    f12_list.extend(df2["F1 CRF"].tolist())
    ttr_list2.extend(df2["TTR"].tolist())
    ucer_list2.extend(df2["UCER"].tolist())

    bleu_list.extend(df3["BleuScore"].tolist())
    f12_list.extend(df3["F1 CRF"].tolist())
    ttr_list2.extend(df3["TTR"].tolist())
    ucer_list2.extend(df3["UCER"].tolist())

    bleu_list.extend(df4["BleuScore"].tolist())
    f12_list.extend(df4["F1 CRF"].tolist())
    ttr_list2.extend(df4["TTR"].tolist())
    ucer_list2.extend(df4["UCER"].tolist())





    trafo_nr = "103"
    path = f"./../experiment_results/rate{trafo_nr}/prob025/all_means.json"
    path2 = f"./../experiment_results/rate{trafo_nr}/all_means.json"
    path3 = f"./../experiment_results/rate{trafo_nr}/prob075/all_means.json"
    path4 = f"./../experiment_results/rate{trafo_nr}/prob1/all_means.json"
    df1 = pd.read_json(path_or_buf=path)
    df2 = pd.read_json(path_or_buf=path2)
    df3 = pd.read_json(path_or_buf=path3)
    df4 = pd.read_json(path_or_buf=path4)
    bleu_list.extend(df1["BleuScore"].tolist())
    f12_list.extend(df1["F1 CRF"].tolist())
    ttr_list2.extend(df1["TTR"].tolist())
    ucer_list2.extend(df1["UCER"].tolist())

    bleu_list.extend(df2["BleuScore"].tolist())
    f12_list.extend(df2["F1 CRF"].tolist())
    ttr_list2.extend(df2["TTR"].tolist())
    ucer_list2.extend(df2["UCER"].tolist())

    bleu_list.extend(df3["BleuScore"].tolist())
    f12_list.extend(df3["F1 CRF"].tolist())
    ttr_list2.extend(df3["TTR"].tolist())
    ucer_list2.extend(df3["UCER"].tolist())

    bleu_list.extend(df4["BleuScore"].tolist())
    f12_list.extend(df4["F1 CRF"].tolist())
    ttr_list2.extend(df4["TTR"].tolist())
    ucer_list2.extend(df4["UCER"].tolist())

    x = np.array(bert_list)
    y = np.array(f1_list)

    # sortiere Indices so, dass x[sorted_indices]
    # das sortierte Array zurückgibt
    sorted_indices = np.argsort(x)

    x = x[sorted_indices]
    y = y[sorted_indices]

    #popt, pcov = curve_fit(f=fn, xdata=x, ydata=y)

    #fit = fn(x, *popt)

    # residual sum of squares
    #ss_res = np.sum((y - fit) ** 2)

    # total sum of squares
    #ss_tot = np.sum((y - np.mean(y)) ** 2)

    # r-squared
    #r2 = 1 - (ss_res / ss_tot)

    #print(r2)


    #cond = np.linalg.cond(pcov)

    print(np.argmax(ttr_list2))
    print(np.argmin(ttr_list2))
    print(len(f12_list))
    fig = plt.scatter(x=ttr_list2, y=f12_list )
    #plt.plot(x, fit, color="#FF1B1B", label="chi²")
    #plt.legend(("p = 0.25", "p = 0.5", "p = 0.75", "p = 1"))
    plt.xlabel("ETTR")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.show()
    figg = fig.figure
    figg.tight_layout()
    # figg.savefig(f"./../experiment_results/trafo{trafo_nr}/plots/scatter_ALL_ettrleonie.pdf")
    # figg.savefig(f"./../experiment_results/trafo{trafo_nr}/plots/scatter_ALL_ettrleonie.png")
    # figg.savefig(f"./../experiment_results/trafo{trafo_nr}/plots/scatter_ALL_ettrleonie.svg")

# Author: Benedikt
def plot_with_diff_rates_rel():
    sns.set_theme()
    str = "86"
    path05 = f"./../experiment_results/rate{str}/all_means.json"
    path025 = f"./../experiment_results/rate{str}/prob025/all_means.json"
    path075 = f"./../experiment_results/rate{str}/prob075/all_means.json"
    path10 = f"./../experiment_results/rate{str}/prob1/all_means.json"

    df = pd.read_json(path_or_buf=path05)
    df2 = pd.read_json(path_or_buf=path025)
    df3 = pd.read_json(path_or_buf=path075)
    df5 = pd.read_json(path_or_buf=path10)
    rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    rate2 = [0.0, 0.3, 0.6, 0.9, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    #rate2 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    df4 = pd.DataFrame({"p = 0.25": df2["F1 Relation"], "p = 0.5": df["F1 Relation"], "p = 0.75": df3["F1 Relation"],
                        "p = 1" : df5["F1 Relation"]})
    df4.index = rate

    min = np.min([np.min(df4["p = 0.5"]), np.min(df4["p = 0.25"]), np.min(df4["p = 0.75"]), np.min(df4["p = 1"])])
    #print(df4)
    # for index, row in df4.iterrows():
    #
    #     x = df4["Prob = 0.5"][index]
    #     df4["Prob = 0.5"][index] = x -min
    # for index, row in df4.iterrows():
    #     x = df4["Prob = 0.25"][index]
    #     df4["Prob = 0.25"][index] = x - min
    # for index, row in df4.iterrows():
    #     x = df4["Prob = 0.75"][index]
    #     df4["Prob = 0.75"][index] = x - min
    # for index, row in df4.iterrows():
    #     x = df4["Prob = 1"][index]
    #     df4["Prob = 1"][index] = x - min

    x = np.linspace(0, 10, 5000)
    not_rate = [0.10, 0.20, 0.40, 0.50,  0.70, 0.80, 1.00]
    #not_rate = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.25, 1.50, 1.75]
    df4 = df4.drop(index=not_rate)

    palette = sns.color_palette(["#FFAEAE", "#92A8FF", "#78D88B", "#E8D381" ],n_colors=4, desat=1)
    fig = sns.lineplot(data=df4, palette=palette)
    fig.set(xlabel="Augmentierungsrate", ylabel="F1 Score")


    popt, pcov = curve_fit(f=fn, xdata=rate2, ydata=df4["p = 0.25"])
    popt2, pcov2 = curve_fit(f=fn, xdata=rate2, ydata=df4["p = 0.5"])
    popt3, pcov3 = curve_fit(f=fn, xdata=rate2, ydata=df4["p = 0.75"])
    popt4, pcov4 = curve_fit(f=fn, xdata=rate2, ydata=df4["p = 1"])



    # print(np.linalg.cond(pcov))
    # print(np.linalg.cond(pcov2))
    # print(np.linalg.cond(pcov3))
    # print(np.linalg.cond(pcov4))


    fit = fn(rate2, *popt)
    fit2 = fn(rate2, *popt2)
    fit3 = fn(rate2, *popt3)
    fit4 = fn(rate2, *popt4)

    #goodness1 = scipy.stats.goodness_of_fit(fit, df4["p = 0.25"])
    #print(goodness1)
    max_x = fit.argmax(axis=0)
    max_x2 = fit2.argmax(axis=0)
    max_x3 = fit3.argmax(axis=0)
    max_x4 = fit4.argmax(axis=0)

    best_aug = rate2[max_x]
    best_aug2 = rate2[max_x2]
    best_aug3 = rate2[max_x3]
    best_aug4 = rate2[max_x4]

    df_best_rate = pd.DataFrame()
    series_rate = pd.Series([best_aug, df4["p = 0.25"][best_aug]])
    df_best_rate = df_best_rate.append(series_rate,ignore_index=True)
    series_rate2 = pd.Series([best_aug, df4["p = 0.5"][best_aug2]])
    df_best_rate = df_best_rate.append(series_rate2,ignore_index=True)
    series_rate3 = pd.Series([best_aug, df4["p = 0.75"][best_aug3]])
    df_best_rate = df_best_rate.append(series_rate3,ignore_index=True)
    series_rate4 = pd.Series([best_aug, df4["p = 1"][best_aug4]])
    df_best_rate = df_best_rate.append(series_rate4, ignore_index=True)

    df_best_rate = df_best_rate.set_axis(["Best Aug Rate", "F1 Score"], axis=1)
    df_best_rate.index = ["p = 0.25", "p = 0.5", "p = 0.75", "p = 1"]
    print(df_best_rate)
    df_best_rate.to_json(path_or_buf=f"./../experiment_results/trafo{str}/best_aug_rate_rel.json", indent=4)



    fig.plot(rate2, fit, color="#FF1B1B", label="Chi2 p = 0.25")
    fig.plot(rate2, fit2, color="#1846FD", label="Chi2 p = 0.5")
    fig.plot(rate2, fit3, color="#0EBC30", label="Chi2 p = 0.75")
    fig.plot(rate2, fit4, color="#DFB81A", label="Chi2 p = 1")

    fig.legend()
    #fig.plot(rate, fit, color="black")

    fig.set_title(f"Transformation {str}")

    #fig.plot(x, chi2.pdf(x, 5), color="black")

    figg = fig.figure
    figg.savefig(f"./../experiment_results/trafo{str}/plots/diff_rates_f1_relation.pdf")
    figg.savefig(f"./../experiment_results/trafo{str}/plots/diff_rates_f1_relation.png")
    figg.savefig(f"./../experiment_results/trafo{str}/plots/diff_rates_f1_relation.svg")
    plt.show()


# Author: Benedikt
def fn5(x, c, s, df):
  return s * chi2.pdf(x, 4.5) + c

# Author: Benedikt
def fn2(x, a, b, c):
    x2 = []
    for i in range(len(x)):
        x2.append(a*x[i]**2 + b*x[i] + c)
    return x2

# Author: Benedikt
def fn3(x, a, c):
    x2 = []
    for i in range(len(x)):
        x2.append(- a*np.exp(-x[i]) + c)
    return x2

# Author: Benedikt
def plot_filter():
    sns.set_theme()
    str = "9"
    str2 = "1"
    path = f"./../experiment_results/filter{str}/exp{9}.{1}/all_means.json"
    name = "UCER"

    df = pd.read_json(path_or_buf=path)



    rate = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]


    df4 = pd.DataFrame({"Filter 9": df[name]})

    df4.index = rate
    palette = sns.color_palette(["#FFAEAE", "#92A8FF", "#78D88B", "#E8D381" ],n_colors=4, desat=1)
    fig = sns.lineplot(data=df4, palette=palette)
    fig.set(xlabel="t", ylabel="CETTR")




    a,b,c = np.polyfit(rate, df4["Filter 9"], deg=2)
    fit2 = fn2(rate, a, b, c)
    fit = np.asarray(fit2)
    max_x = fit.argmax(axis=0)
    best_length = rate[max_x]
    print(best_length)
    fig.plot(rate, fit2, color="#FF1B1B", label="x² Fit für Filter 9")
    fig.legend()
    plt.show()
    print(df4)
    figg = fig.figure
    figg.tight_layout()
    figg.savefig(f"./../experiment_results/filter{str}/plots/{name}_length.pdf")
    figg.savefig(f"./../experiment_results/filter{str}/plots/{name}_length.png")
    figg.savefig(f"./../experiment_results/filter{str}/plots/{name}_length.svg")


# Author: Benedikt
def fnf(x, d, c):
    x2 = []
    for i in range(len(x)):
        x2.append(x[i]/(2*x[i] + d)+c)
    return x2

# Author: Benedikt
def plot_filter10_1():
    sns.set_theme()
    str = "19"
    str2 = "1"
    path = f"./../experiment_results/filter10/exp10.1/all_means.json"
    path2 = f"./../experiment_results/filter19/exp19.1/all_means.json"
    name = "F1 CRF"

    df = pd.read_json(path_or_buf=path)
    df2 = pd.read_json(path_or_buf=path2)


    rate = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]


    #df4 = pd.DataFrame({"Filter 10": df[name], "Filter 19": df2[name]})
    df4 = pd.DataFrame({"Filter 19": df2[name]})
    df4.index = rate
    palette = sns.color_palette(["#FFAEAE", "#92A8FF", "#78D88B", "#E8D381" ],n_colors=4, desat=1)
    fig = sns.lineplot(data=df4, palette=palette)
    fig.set(xlabel="t", ylabel="F1 CRF")




   # a,b,c = np.polyfit(rate, df4["Filter 19"], deg=2)
   # c, cov = curve_fit(fn2, xdata=rate, ydata=df4["Filter 10"])
    c2, cov2 = curve_fit(fnf, xdata=rate, ydata=df4["Filter 19"])
    #print(c)
    #fit2 = fn2(rate,*c)
    fit3 = fnf(rate, *c2)
    #print(fit2)
    # fit = np.asarray(fit2)
    # max_x = fit.argmax(axis=0)
    # best_length = rate[max_x]
    # print(best_length)
    #fig.plot(rate, fit2, color="#FF1B1B", label="x² Fit für Filter 10")
    fig.plot(rate, fit3, color="#FF1B1B", label="x/(2x + d)+c Fit für Filter 19")
    #fig.plot(rate, fit3, color="#1846FD", label="x/(2x + d)+c Fit für Filter 19")
    fig.legend()
    plt.show()
    print(df4)
    figg = fig.figure
    figg.tight_layout()
    figg.savefig(f"./../experiment_results/filter{str}/plots/f1_countAll.pdf")
    figg.savefig(f"./../experiment_results/filter{str}/plots/f1_countAll.png")
    figg.savefig(f"./../experiment_results/filter{str}/plots/f1_countAll.svg")


# Author: Benedikt
def plot_filter10_3():
    sns.set_theme()
    str = "10"
    str2 = "3"
    path = f"./../experiment_results/filter10/exp10.1/all_means.json"
    path2 = f"./../experiment_results/filter10/exp10.1/all_means.json"
    name = "BertScore"

    df = pd.read_json(path_or_buf=path)
    df2 = pd.read_json(path_or_buf=path2)


    rate = ["Actor", "Activity", "Act. Data", "Fur. Spec.", "XOR", "Cond. Spec.", "AND"]
    #rate = ["Nomen", "Adjektive", "Verben"]

    #df4 = pd.DataFrame({"Filter 10": df["F1 CRF"]})
    df4 = pd.DataFrame(data=df[name])
    df4.index = rate
    palette = sns.color_palette(["#FFAEAE", "#92A8FF", "#78D88B", "#E8D381" ],n_colors=4, desat=1)
    fig = sns.barplot(x=rate, y=df4[name])

    #fig.set(xlabel="Entity", ylabel="F1 Score")




    # a,b,c = np.polyfit(rate, df4["Filter 10"], deg=2)
    # fit2 = fn2(rate, a, b, c)
    # fit = np.asarray(fit2)
    # max_x = fit.argmax(axis=0)
    # best_length = rate[max_x]
    # print(best_length)
    # fig.plot(rate, fit2, color="#FF1B1B", label="x² Fit für Filter 10")
    #fig.legend()
    plt.show()
    print(df4)
    figg = fig.figure
    figg.tight_layout()
    figg.savefig(f"./../experiment_results/filter{str}/plots/bert_pos.pdf")
    figg.savefig(f"./../experiment_results/filter{str}/plots/bert_pos.png")
    figg.savefig(f"./../experiment_results/filter{str}/plots/bert_pos.svg")


# Author: Benedikt
def scatter_filter():
    sns.set_theme()
    str = "19"
    str2 = "1"
    path = f"./../experiment_results/trafo58/exp58.1/all_means.json"
    #path2 = f"./../experiment_results/filter58/58.1/all_means.json"

    df =  pd.read_json(path_or_buf=path)
    #df2 = pd.read_json(path_or_buf=path2)
    #df4 = pd.DataFrame({"Filter 10": })
    #fig = sns.lmplot(x='BertScore', y='F1 CRF', data=df, fit_reg=True )
    fig = sns.lmplot(x='Bert Score', y='F1 Score', data=df, fit_reg=True)
    plt.show()
    figg = fig.figure
    figg.tight_layout()
    figg.savefig(f"./../experiment_results/trafo58/plots/scatter_fbert.pdf")
    figg.savefig(f"./../experiment_results/trafo58/plots/scatter_fbert.png")
    figg.savefig(f"./../experiment_results/trafo58/plots/scatter_fbert.svg")




scatter()