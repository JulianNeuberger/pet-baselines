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

def plot_train_as_test():
    sns.set_theme()
    str = "101"
    path = f"./../experiment_results/rate3/test_all_means.json"
    path2 = f"./../experiment_results/rate39/test_all_means.json"
    path3 = f"./../experiment_results/rate40/test_all_means.json"
    path4 = f"./../experiment_results/rate86/test_all_means.json"
    path5 = f"./../experiment_results/rate90/test_all_means.json"
    path6 = f"./../experiment_results/rate103/test_all_means.json"

    df = pd.read_json(path_or_buf=path)
    df2 = pd.read_json(path_or_buf=path2)
    df3 = pd.read_json(path_or_buf=path3)
    df5 = pd.read_json(path_or_buf=path4)
    df6 = pd.read_json(path_or_buf=path5)
    df7 = pd.read_json(path_or_buf=path6)
    rate = [0.0, 1.0, 2.0, 3.0, 4.0]
    df4 = pd.DataFrame({"Trafo 3": df["F1 CRF"], "Trafo 39": df2["F1 CRF"], "Trafo 40": df3["F1 CRF"],
                        "Trafo 86": df5["F1 CRF"],"Trafo 90": df6["F1 CRF"],"Trafo 103": df7["F1 CRF"]})
    df4.index = rate

    fig = sns.lineplot(data=df4)
    fig.set(xlabel="Augmentierungsrate", ylabel="F1 Score")

    fig.set_title("Augmented Test Data")
    figg = fig.figure
    figg.savefig("./../experiment_results/Plots/train_as_testLeonie.pdf")
    figg.savefig("./../experiment_results/Plots/train_as_testLeonie.png")
    figg.savefig("./../experiment_results/Plots/train_as_testLeonie.svg")
    plt.show()


def fn(x, c, s, df):
  return s * chi2.pdf(x, 4.5) + c


def plot_with_diff_rates():
    sns.set_theme()
    str = "101"
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
    df4 = pd.DataFrame({"p = 0.25": df2["F1 CRF"], "p = 0.5": df["F1 CRF"], "p = 0.75": df3["F1 CRF"],
                        "p = 1" : df5["F1 CRF"]})
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
    df_best_rate.to_json(path_or_buf=f"./../experiment_results/trafo{str}/best_aug_rate.json", indent=4)



    fig.plot(rate2, fit, color="#FF1B1B", label="Chi2 p = 0.25")
    fig.plot(rate2, fit2, color="#1846FD", label="Chi2 p = 0.5")
    fig.plot(rate2, fit3, color="#0EBC30", label="Chi2 p = 0.75")
    fig.plot(rate2, fit4, color="#DFB81A", label="Chi2 p = 1")

    fig.legend()
    #fig.plot(rate, fit, color="black")

    fig.set_title(f"Transformation {str}")

    #fig.plot(x, chi2.pdf(x, 5), color="black")

    figg = fig.figure
    figg.savefig(f"./../experiment_results/trafo{str}/plots/diff_rates_f1.pdf")
    figg.savefig(f"./../experiment_results/trafo{str}/plots/diff_rates_f1.png")
    figg.savefig(f"./../experiment_results/trafo{str}/plots/diff_rates_f1.svg")
    plt.show()



def get_df_with_all_ttr_means_per_prob_rate(trafo_nr):
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
    new_series4 = pd.Series(list1)
    df_all_ttr_means = df_all_ttr_means.append(new_series4, ignore_index=True)

    list2 = []
    for i in rate:
        path = f"./../experiment_results/rate{trafo_nr}/prob075/ttr_mean_un_{i}.json"
        df = pd.read_json(path_or_buf=path)
        list2.append(df["All"][0])
    new_series2 = pd.Series(list1)
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
    new_series4 = pd.Series(list1)
    df_all_ttr_means = df_all_ttr_means.append(new_series4, ignore_index=True)

    list2 = []
    for i in rate:
        path = f"./../experiment_results/rate{trafo_nr}/prob075/ucer_mean_un_{i}.json"
        df = pd.read_json(path_or_buf=path)
        list2.append(df["All"][0])
    new_series2 = pd.Series(list1)
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

    df_new_all_means_aug_ttr = df_new_all_means_aug_ttr.set_axis([0.25, 0.5, 0.75, 1.0], axis = 1)
    df_new_all_means_aug_ttr.to_json(
        path_or_buf=f"./../experiment_results/rate{trafo_nr}/aaa_bleu_means_per_prob_and_rate.json", indent=4)


def plot_ttr_means_per_prob_and_rate_ratetottr(trafo_nr):
    sns.set_theme()
    #trafo_nr = "39"
    rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]

    path = f"./../experiment_results/rate{trafo_nr}/aaa_ttr_means_un_per_prob_and_rate.json"
    df1 = pd.read_json(path_or_buf=path)
    path2 = f"./../experiment_results/rate{trafo_nr}/aaa_ttr_means_per_prob_and_rate.json"
    df2 = pd.read_json(path_or_buf=path2)

    df2 = df2.subtract(df1)
    df2 = df2.set_axis(rate, axis=1)
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


def plot_ttr_means_per_prob_and_rate_ttrtof1():
    trafo_nr = "3"

    path = f"./../experiment_results/rate{trafo_nr}/aaa_ttr_means_un_per_prob_and_rate.json"
    df = pd.read_json(path_or_buf=path)






# for i in [3, 5, 39, 40, 82, 86, 90, 100, 101, 103]:
#     get_df_with_all_ttr_means_per_prob_rate(i)
#     #plot_ttr_means_per_prob_and_rate_ratetottr(i)
#     #pass
#
#
#
# for i in [3, 5, 39, 40, 82, 86, 90, 100, 101, 103]:
#    get_df_with_all_ucer_means_per_prob_rate(i)
#     #plot_ucer_means_per_prob_and_rate_ratetottr(i)
#     #pass
#
#
# for i in [3, 39, 40, 86, 90, 103]:
#     get_df_with_all_bleu_means_per_prob_rate(i)
#     #plot_ucer_means_per_prob_and_rate_ratetottr(i)
#     #pass
plot_with_diff_rates()
