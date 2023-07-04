import json
import typing
import os
import data
import augment
import pipeline
import main as console
from transformations import tokenmanager
import pandas as pd
import main
import tqdm
from metrics_ba import Metrics


def run_experiment(name: str, aug_train_folds, test_folds):
    print(f'building scores for {name}')
    crf_ext = pipeline.CrfMentionEstimatorStep(name='crf mention extraction')
    neural_ext = pipeline.NeuralCoReferenceResolutionStep(name='neural coreference resolution',
                                                     resolved_tags=['Actor', 'Activity Data'],
                                                     cluster_overlap=.5,
                                                     mention_overlap=.8,
                                                     ner_strategy='frequency')
    rel_ext_rule = pipeline.RuleBasedRelationExtraction(name='rule-based relation extraction')
    #rel_ext = pipeline.CatBoostRelationExtractionStep(name='perfect entities', context_size=2,
                                                    #num_trees=100, negative_sampling_rate=40.0)
    res = cross_validate_pipeline_macro(
        p=pipeline.Pipeline(name=name, steps=[crf_ext, neural_ext, rel_ext_rule]),
        train_folds=aug_train_folds,
        test_folds=test_folds
    )
    scores_crf = res[crf_ext]
    scores_neural = res[neural_ext]
    scores_rel_rule = res[rel_ext_rule]
    #scores_rel = res[rel_ext]

    #scores = list(res.values())[0]
    f_score_crf = scores_crf.overall_scores.f1
    f_score_neural = scores_neural.overall_scores.f1
    f_score_rel_rule = scores_rel_rule.overall_scores.f1
    #f_score_rel = scores_rel.overall_scores.f1

    new_list = {}
    new_list["Actor"] = scores_crf.scores_by_tag["actor"].f1
    new_list["Activity"] = scores_crf.scores_by_tag["activity"].f1
    new_list["Activity Data"] = scores_crf.scores_by_tag["activity data"].f1
    new_list["Further Specification"] = scores_crf.scores_by_tag["further specification"].f1
    new_list["XOR Gateway"] = scores_crf.scores_by_tag["xor gateway"].f1
    new_list["Condition Specification"] = scores_crf.scores_by_tag["condition specification"].f1
    new_list["AND Gateway"] = scores_crf.scores_by_tag["and gateway"].f1

    new_series = pd.Series(data=new_list)
    return [f_score_crf, f_score_neural, f_score_rel_rule, new_series]
    #return [f_score_crf, f_score_neural, f_score_rel, new_series]


def evaluate_unaugmented_data(unaug_train_folds, aug_train_folds, f_score):
    metrics: Metrics = Metrics(unaug_train_set=unaug_train_folds, train_set=aug_train_folds)
    df_ttr = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                   'Condition Specification', 'AND Gateway', 'All'])
    df_ttr_mean = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                        'Condition Specification', 'AND Gateway', 'All'])
    df_ucer = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                    'Condition Specification', 'AND Gateway', 'All'])
    df_ucer_mean = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                         'Condition Specification', 'AND Gateway', 'All'])

    for i in range(5):
        # get TTR
        new_df_ttr_series = metrics.calculate_ttr(i)
        df_ttr = df_ttr.append(new_df_ttr_series, ignore_index=True)

        # get UCER
        new_df_ucer_series = metrics.calculate_ttr_trigram(i)
        df_ucer = df_ucer.append(new_df_ucer_series, ignore_index=True)

    # get mean ttr
    new_df_ttr_mean_dict = {
        'Actor': df_ttr["Actor"].mean(),
        'Activity': df_ttr["Activity"].mean(),
        'Activity Data': df_ttr["Activity Data"].mean(),
        'Further Specification': df_ttr["Further Specification"].mean(),
        'XOR Gateway': df_ttr["XOR Gateway"].mean(),
        'Condition Specification': df_ttr["Condition Specification"].mean(),
        'AND Gateway': df_ttr["AND Gateway"].mean(),
        'All': df_ttr["All"].mean(),
    }
    new_df_ttr_mean_series = pd.Series(new_df_ttr_mean_dict)
    df_ttr_mean = df_ttr_mean.append(new_df_ttr_mean_series, ignore_index=True)

    # get mean ucer
    new_df_ucer_mean_dict = {
        'Actor': df_ucer["Actor"].mean(),
        'Activity': df_ucer["Activity"].mean(),
        'Activity Data': df_ucer["Activity Data"].mean(),
        'Further Specification': df_ucer["Further Specification"].mean(),
        'XOR Gateway': df_ucer["XOR Gateway"].mean(),
        'Condition Specification': df_ucer["Condition Specification"].mean(),
        'AND Gateway': df_ucer["AND Gateway"].mean(),
        'All': df_ucer["All"].mean(),
    }
    new_df_ucer_mean_series = pd.Series(new_df_ucer_mean_dict)
    df_ucer_mean = df_ucer_mean.append(new_df_ucer_mean_series, ignore_index=True)


    # get df complete series
    new_dict_series = {
        '$F_{1}$': f_score,
        '$TTR$': df_ttr_mean['All'][0],
        '$UCER$': df_ucer_mean['All'][0]}
    new_df_complete_series = pd.Series(data=new_dict_series)

    return new_df_complete_series, df_ttr, df_ucer, df_ttr_mean, df_ucer_mean


def evaluate_experiment_bleu(unaug_train_folds, aug_train_folds, f_score_crf, f_score_neural, f_score_rel):
    metrics: Metrics = Metrics(unaug_train_set=unaug_train_folds, train_set=aug_train_folds)
    df_ttr = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                   'Condition Specification', 'AND Gateway', 'All'])
    df_ttr_mean = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                   'Condition Specification', 'AND Gateway', 'All'])
    df_ucer = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                   'Condition Specification', 'AND Gateway', 'All'])
    df_ucer_mean = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                   'Condition Specification', 'AND Gateway', 'All'])
    df_bleu = pd.DataFrame(columns=['Bleu Score'])

    for i in range(5):
        # get TTR
        new_df_ttr_series = metrics.calculate_ttr(i)
        df_ttr = df_ttr.append(new_df_ttr_series, ignore_index=True)

        # get UCER
        new_df_ucer_series = metrics.calculate_ttr_trigram(i)
        df_ucer = df_ucer.append(new_df_ucer_series, ignore_index=True)

        # get Bleu
        new_df_bleu_series = metrics.calculate_bleu(i)
        df_bleu = df_bleu.append(new_df_bleu_series, ignore_index=True)

    # get mean ttr
    new_df_ttr_mean_dict = {
        'Actor': df_ttr["Actor"].mean(),
        'Activity': df_ttr["Activity"].mean(),
        'Activity Data': df_ttr["Activity Data"].mean(),
        'Further Specification': df_ttr["Further Specification"].mean(),
        'XOR Gateway': df_ttr["XOR Gateway"].mean(),
        'Condition Specification': df_ttr["Condition Specification"].mean(),
        'AND Gateway': df_ttr["AND Gateway"].mean(),
        'All': df_ttr["All"].mean(),
    }
    new_df_ttr_mean_series = pd.Series(new_df_ttr_mean_dict)
    df_ttr_mean = df_ttr_mean.append(new_df_ttr_mean_series, ignore_index=True)

    # get mean ucer
    new_df_ucer_mean_dict = {
        'Actor': df_ucer["Actor"].mean(),
        'Activity': df_ucer["Activity"].mean(),
        'Activity Data': df_ucer["Activity Data"].mean(),
        'Further Specification': df_ucer["Further Specification"].mean(),
        'XOR Gateway': df_ucer["XOR Gateway"].mean(),
        'Condition Specification': df_ucer["Condition Specification"].mean(),
        'AND Gateway': df_ucer["AND Gateway"].mean(),
        'All': df_ucer["All"].mean(),
    }
    new_df_ucer_mean_series = pd.Series(new_df_ucer_mean_dict)
    df_ucer_mean = df_ucer_mean.append(new_df_ucer_mean_series, ignore_index=True)

    # get mean bleu
    bleu_mean = df_bleu['Bleu Score'].mean()

    # get df complete series
    new_dict_series = {
         'F1 CRF': f_score_crf,
         'F1 Neural': f_score_neural,
         'F1 Relation': f_score_rel,
         'TTR': df_ttr_mean['All'][0],
         'UCER': df_ucer_mean['All'][0],
         'BleuScore': bleu_mean}
    new_df_complete_series = pd.Series(data=new_dict_series)

    return new_df_complete_series, df_ttr, df_ucer, df_ttr_mean, df_ucer_mean, df_bleu, bleu_mean


def evaluate_experiment_bert(unaug_train_folds, aug_train_folds, f_score_crf, f_score_neural, f_score_rel):
    metrics: Metrics = Metrics(unaug_train_set=unaug_train_folds, train_set=aug_train_folds)
    df_ttr = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                   'Condition Specification', 'AND Gateway', 'All'])
    df_ttr_mean = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                        'Condition Specification', 'AND Gateway', 'All'])
    df_ucer = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                    'Condition Specification', 'AND Gateway', 'All'])
    df_ucer_mean = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                         'Condition Specification', 'AND Gateway', 'All'])
    df_bert = pd.DataFrame(columns=['Bert Score'])

    for i in range(5):
        # get TTR
        new_df_ttr_series = metrics.calculate_ttr(i)
        df_ttr = df_ttr.append(new_df_ttr_series, ignore_index=True)

        # get UCER
        new_df_ucer_series = metrics.calculate_ttr_trigram(i)
        df_ucer = df_ucer.append(new_df_ucer_series, ignore_index=True)

        # get Bert
        new_df_bert_series = metrics.calculate_bert(i)
        df_bert = df_bert.append(new_df_bert_series, ignore_index=True)

    # get mean ttr
    new_df_ttr_mean_dict = {
        'Actor': df_ttr["Actor"].mean(),
        'Activity': df_ttr["Activity"].mean(),
        'Activity Data': df_ttr["Activity Data"].mean(),
        'Further Specification': df_ttr["Further Specification"].mean(),
        'XOR Gateway': df_ttr["XOR Gateway"].mean(),
        'Condition Specification': df_ttr["Condition Specification"].mean(),
        'AND Gateway': df_ttr["AND Gateway"].mean(),
        'All': df_ttr["All"].mean(),
    }
    new_df_ttr_mean_series = pd.Series(new_df_ttr_mean_dict)
    df_ttr_mean = df_ttr_mean.append(new_df_ttr_mean_series, ignore_index=True)

    # get mean ucer
    new_df_ucer_mean_dict = {
        'Actor': df_ucer["Actor"].mean(),
        'Activity': df_ucer["Activity"].mean(),
        'Activity Data': df_ucer["Activity Data"].mean(),
        'Further Specification': df_ucer["Further Specification"].mean(),
        'XOR Gateway': df_ucer["XOR Gateway"].mean(),
        'Condition Specification': df_ucer["Condition Specification"].mean(),
        'AND Gateway': df_ucer["AND Gateway"].mean(),
        'All': df_ucer["All"].mean(),
    }
    new_df_ucer_mean_series = pd.Series(new_df_ucer_mean_dict)
    df_ucer_mean = df_ucer_mean.append(new_df_ucer_mean_series, ignore_index=True)
    print(df_bert)
    # get mean bleu
    bert_mean = df_bert['Bert Score'].mean()

    # get df complete series
    new_dict_series = {
        'F1 CRF': f_score_crf,
         'F1 Neural': f_score_neural,
         'F1 Relation': f_score_rel,
        'TTR': df_ttr_mean['All'][0],
        'UCER': df_ucer_mean['All'][0],
        'BertScore': bert_mean}
    new_df_complete_series = pd.Series(data=new_dict_series)

    return new_df_complete_series, df_ttr, df_ucer, df_ttr_mean, df_ucer_mean, df_bert, bert_mean


def cross_validate_pipeline_macro(p: pipeline.Pipeline, *,
                            train_folds: typing.List[typing.List[data.Document]],
                            test_folds: typing.List[typing.List[data.Document]],
                            save_results: bool = False,
                            dump_predictions_dir: str = None):
    assert len(train_folds) == len(test_folds)
    pipeline_results = []
    for n_fold, (train_fold, test_fold) in tqdm.tqdm(enumerate(zip(train_folds, test_folds)),
                                                     total=len(train_folds), desc='cross validation fold'):
        ground_truth = [d.copy() for d in test_fold]
        pipeline_result = p.run(train_documents=train_fold,
                                test_documents=test_fold,
                                ground_truth_documents=ground_truth)
        pipeline_results.append(pipeline_result)
    res = main.accumulate_pipeline_results(pipeline_results, averaging_mode="macro")
    if dump_predictions_dir is not None:
        for i, pipeline_result in enumerate(pipeline_results):
            json_data = [p.to_json_serializable() for p in pipeline_result.step_results[p.steps[-1]].predictions]
            os.makedirs(dump_predictions_dir, exist_ok=True)
            with open(os.path.join(dump_predictions_dir, f'fold-{i}.json'), 'w', encoding='utf8') as f:
                json.dump(json_data, f)

    if save_results:
        df_persistence = 'experiments.pkl'
        if os.path.isfile(df_persistence):
            df = pd.read_pickle(df_persistence)
        else:
            df = pd.DataFrame(columns=['experiment_name', 'tag', 'p', 'r', 'f1'])
        final_result = res[p.steps[-1]]
        for tag, value in final_result.scores_by_tag.items():
            df = df.append({
                'experiment_name': p.name,
                'tag': tag,
                'p': value.p,
                'r': value.r,
                'f1': value.f1
            }, ignore_index=True)
        df = df.append({
            'experiment_name': p.name,
            'tag': 'overall',
            'p': final_result.overall_scores.p,
            'r': final_result.overall_scores.r,
            'f1': final_result.overall_scores.f1
        }, ignore_index=True)
        pd.to_pickle(df, df_persistence)

    main.print_pipeline_results(p, res)
    return res


def evaluate_experiment_with_rate(unaug_train_folds, aug_train_folds, f_score_crf, f_score_neural, f_score_rel):
    metrics: Metrics = Metrics(unaug_train_set=unaug_train_folds, train_set=aug_train_folds)
    df_ttr = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                   'Condition Specification', 'AND Gateway', 'All'])
    df_ttr_mean = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                        'Condition Specification', 'AND Gateway', 'All'])
    df_ucer = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                    'Condition Specification', 'AND Gateway', 'All'])
    df_ucer_mean = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                         'Condition Specification', 'AND Gateway', 'All'])
    df_bert = pd.DataFrame(columns=['Bert Score'])
    df_ttr_un = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                   'Condition Specification', 'AND Gateway', 'All'])
    df_ttr_mean_un = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                        'Condition Specification', 'AND Gateway', 'All'])
    df_ucer_un = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                    'Condition Specification', 'AND Gateway', 'All'])
    df_ucer_mean_un = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                         'Condition Specification', 'AND Gateway', 'All'])


    for i in range(5):
        # get TTR
        new_df_ttr_series = metrics.calculate_ttr(i)
        df_ttr = df_ttr.append(new_df_ttr_series, ignore_index=True)

        # get UCER
        new_df_ucer_series = metrics.calculate_ttr_trigram(i)
        df_ucer = df_ucer.append(new_df_ucer_series, ignore_index=True)

        # get Bert
        new_df_bert_series = metrics.calculate_bert(i)
        df_bert = df_bert.append(new_df_bert_series, ignore_index=True)

        # get TTR
        new_df_ttr_series_un = metrics.calculate_ttr_un(i)
        df_ttr_un = df_ttr_un.append(new_df_ttr_series_un, ignore_index=True)

        # get UCER
        new_df_ucer_series_un = metrics.calculate_ttr_trigram_un(i)
        df_ucer_un = df_ucer_un.append(new_df_ucer_series_un, ignore_index=True)

    # get mean ttr
    new_df_ttr_mean_dict = {
        'Actor': df_ttr["Actor"].mean(),
        'Activity': df_ttr["Activity"].mean(),
        'Activity Data': df_ttr["Activity Data"].mean(),
        'Further Specification': df_ttr["Further Specification"].mean(),
        'XOR Gateway': df_ttr["XOR Gateway"].mean(),
        'Condition Specification': df_ttr["Condition Specification"].mean(),
        'AND Gateway': df_ttr["AND Gateway"].mean(),
        'All': df_ttr["All"].mean(),
    }
    new_df_ttr_mean_series = pd.Series(new_df_ttr_mean_dict)
    df_ttr_mean = df_ttr_mean.append(new_df_ttr_mean_series, ignore_index=True)

    # get mean ttr
    new_df_ttr_mean_dict_un = {
        'Actor': df_ttr_un["Actor"].mean(),
        'Activity': df_ttr_un["Activity"].mean(),
        'Activity Data': df_ttr_un["Activity Data"].mean(),
        'Further Specification': df_ttr_un["Further Specification"].mean(),
        'XOR Gateway': df_ttr_un["XOR Gateway"].mean(),
        'Condition Specification': df_ttr_un["Condition Specification"].mean(),
        'AND Gateway': df_ttr_un["AND Gateway"].mean(),
        'All': df_ttr_un["All"].mean(),
    }
    new_df_ttr_mean_series_un = pd.Series(new_df_ttr_mean_dict_un)
    df_ttr_mean_un = df_ttr_mean_un.append(new_df_ttr_mean_series_un, ignore_index=True)

    # get mean ucer
    new_df_ucer_mean_dict = {
        'Actor': df_ucer["Actor"].mean(),
        'Activity': df_ucer["Activity"].mean(),
        'Activity Data': df_ucer["Activity Data"].mean(),
        'Further Specification': df_ucer["Further Specification"].mean(),
        'XOR Gateway': df_ucer["XOR Gateway"].mean(),
        'Condition Specification': df_ucer["Condition Specification"].mean(),
        'AND Gateway': df_ucer["AND Gateway"].mean(),
        'All': df_ucer["All"].mean(),
    }
    new_df_ucer_mean_series = pd.Series(new_df_ucer_mean_dict)
    df_ucer_mean = df_ucer_mean.append(new_df_ucer_mean_series, ignore_index=True)

    # get mean ucer
    new_df_ucer_mean_dict_un = {
        'Actor': df_ucer_un["Actor"].mean(),
        'Activity': df_ucer_un["Activity"].mean(),
        'Activity Data': df_ucer_un["Activity Data"].mean(),
        'Further Specification': df_ucer_un["Further Specification"].mean(),
        'XOR Gateway': df_ucer_un["XOR Gateway"].mean(),
        'Condition Specification': df_ucer_un["Condition Specification"].mean(),
        'AND Gateway': df_ucer_un["AND Gateway"].mean(),
        'All': df_ucer_un["All"].mean(),
    }
    new_df_ucer_mean_series_un = pd.Series(new_df_ucer_mean_dict_un)
    df_ucer_mean_un = df_ucer_mean_un.append(new_df_ucer_mean_series_un, ignore_index=True)

    bert_mean = df_bert['Bert Score'].mean()
    # get df complete series
    new_dict_series = {
        'F1 CRF': f_score_crf,
         'F1 Neural': f_score_neural,
         'F1 Relation': f_score_rel,
        'TTR': df_ttr_mean['All'][0],
        'UCER': df_ucer_mean['All'][0],
        'BertScore': bert_mean}
    new_df_complete_series = pd.Series(data=new_dict_series)

    return new_df_complete_series, df_ttr, df_ucer, df_ttr_mean, df_ucer_mean, df_bert, df_ttr_un, df_ucer_un, \
        df_ttr_mean_un, df_ucer_mean_un, bert_mean


def evaluate_experiment_with_rate_bleu(unaug_train_folds, aug_train_folds, f_score_crf, f_score_neural, f_score_rel):
    metrics: Metrics = Metrics(unaug_train_set=unaug_train_folds, train_set=aug_train_folds)
    df_ttr = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                   'Condition Specification', 'AND Gateway', 'All'])
    df_ttr_mean = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                        'Condition Specification', 'AND Gateway', 'All'])
    df_ucer = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                    'Condition Specification', 'AND Gateway', 'All'])
    df_ucer_mean = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                         'Condition Specification', 'AND Gateway', 'All'])
    df_bleu = pd.DataFrame(columns=['Bleu Score'])
    df_ttr_un = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                   'Condition Specification', 'AND Gateway', 'All'])
    df_ttr_mean_un = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                        'Condition Specification', 'AND Gateway', 'All'])
    df_ucer_un = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                    'Condition Specification', 'AND Gateway', 'All'])
    df_ucer_mean_un = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                         'Condition Specification', 'AND Gateway', 'All'])


    for i in range(5):
        # get TTR
        new_df_ttr_series = metrics.calculate_ttr(i)
        df_ttr = df_ttr.append(new_df_ttr_series, ignore_index=True)

        # get UCER
        new_df_ucer_series = metrics.calculate_ttr_trigram(i)
        df_ucer = df_ucer.append(new_df_ucer_series, ignore_index=True)

        # get Bleu
        new_df_bleu_series = metrics.calculate_bleu_rate(i)
        df_bleu = df_bleu.append(new_df_bleu_series, ignore_index=True)

        # get TTR
        new_df_ttr_series_un = metrics.calculate_ttr_un(i)
        df_ttr_un = df_ttr_un.append(new_df_ttr_series_un, ignore_index=True)

        # get UCER
        new_df_ucer_series_un = metrics.calculate_ttr_trigram_un(i)
        df_ucer_un = df_ucer_un.append(new_df_ucer_series_un, ignore_index=True)

    # get mean ttr
    new_df_ttr_mean_dict = {
        'Actor': df_ttr["Actor"].mean(),
        'Activity': df_ttr["Activity"].mean(),
        'Activity Data': df_ttr["Activity Data"].mean(),
        'Further Specification': df_ttr["Further Specification"].mean(),
        'XOR Gateway': df_ttr["XOR Gateway"].mean(),
        'Condition Specification': df_ttr["Condition Specification"].mean(),
        'AND Gateway': df_ttr["AND Gateway"].mean(),
        'All': df_ttr["All"].mean(),
    }
    new_df_ttr_mean_series = pd.Series(new_df_ttr_mean_dict)
    df_ttr_mean = df_ttr_mean.append(new_df_ttr_mean_series, ignore_index=True)

    # get mean ttr
    new_df_ttr_mean_dict_un = {
        'Actor': df_ttr_un["Actor"].mean(),
        'Activity': df_ttr_un["Activity"].mean(),
        'Activity Data': df_ttr_un["Activity Data"].mean(),
        'Further Specification': df_ttr_un["Further Specification"].mean(),
        'XOR Gateway': df_ttr_un["XOR Gateway"].mean(),
        'Condition Specification': df_ttr_un["Condition Specification"].mean(),
        'AND Gateway': df_ttr_un["AND Gateway"].mean(),
        'All': df_ttr_un["All"].mean(),
    }
    new_df_ttr_mean_series_un = pd.Series(new_df_ttr_mean_dict_un)
    df_ttr_mean_un = df_ttr_mean_un.append(new_df_ttr_mean_series_un, ignore_index=True)

    # get mean ucer
    new_df_ucer_mean_dict = {
        'Actor': df_ucer["Actor"].mean(),
        'Activity': df_ucer["Activity"].mean(),
        'Activity Data': df_ucer["Activity Data"].mean(),
        'Further Specification': df_ucer["Further Specification"].mean(),
        'XOR Gateway': df_ucer["XOR Gateway"].mean(),
        'Condition Specification': df_ucer["Condition Specification"].mean(),
        'AND Gateway': df_ucer["AND Gateway"].mean(),
        'All': df_ucer["All"].mean(),
    }
    new_df_ucer_mean_series = pd.Series(new_df_ucer_mean_dict)
    df_ucer_mean = df_ucer_mean.append(new_df_ucer_mean_series, ignore_index=True)

    # get mean ucer
    new_df_ucer_mean_dict_un = {
        'Actor': df_ucer_un["Actor"].mean(),
        'Activity': df_ucer_un["Activity"].mean(),
        'Activity Data': df_ucer_un["Activity Data"].mean(),
        'Further Specification': df_ucer_un["Further Specification"].mean(),
        'XOR Gateway': df_ucer_un["XOR Gateway"].mean(),
        'Condition Specification': df_ucer_un["Condition Specification"].mean(),
        'AND Gateway': df_ucer_un["AND Gateway"].mean(),
        'All': df_ucer_un["All"].mean(),
    }
    new_df_ucer_mean_series_un = pd.Series(new_df_ucer_mean_dict_un)
    df_ucer_mean_un = df_ucer_mean_un.append(new_df_ucer_mean_series_un, ignore_index=True)

    bleu_mean = df_bleu['Bleu Score'].mean()
    # get df complete series
    new_dict_series = {
        'F1 CRF': f_score_crf,
         'F1 Neural': f_score_neural,
         'F1 Relation': f_score_rel,
        'TTR': df_ttr_mean['All'][0],
        'UCER': df_ucer_mean['All'][0],
        'BleuScore': bleu_mean}
    new_df_complete_series = pd.Series(data=new_dict_series)

    return new_df_complete_series, df_ttr, df_ucer, df_ttr_mean, df_ucer_mean, df_bleu, df_ttr_un, df_ucer_un, \
        df_ttr_mean_un, df_ucer_mean_un, bleu_mean

def evaluate_experiment_test(unaug_train_folds, aug_train_folds, f_score_crf, f_score_neural, f_score_rel):
    metrics: Metrics = Metrics(unaug_train_set=unaug_train_folds, train_set=aug_train_folds)
    df_ttr = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                   'Condition Specification', 'AND Gateway', 'All'])
    df_ttr_mean = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                        'Condition Specification', 'AND Gateway', 'All'])
    df_ucer = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                    'Condition Specification', 'AND Gateway', 'All'])
    df_ucer_mean = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                         'Condition Specification', 'AND Gateway', 'All'])


    for i in range(5):
        # get TTR
        new_df_ttr_series = metrics.calculate_ttr(i)
        df_ttr = df_ttr.append(new_df_ttr_series, ignore_index=True)

        # get UCER
        new_df_ucer_series = metrics.calculate_ttr_trigram(i)
        df_ucer = df_ucer.append(new_df_ucer_series, ignore_index=True)

        # get Bert


    # get mean ttr
    new_df_ttr_mean_dict = {
        'Actor': df_ttr["Actor"].mean(),
        'Activity': df_ttr["Activity"].mean(),
        'Activity Data': df_ttr["Activity Data"].mean(),
        'Further Specification': df_ttr["Further Specification"].mean(),
        'XOR Gateway': df_ttr["XOR Gateway"].mean(),
        'Condition Specification': df_ttr["Condition Specification"].mean(),
        'AND Gateway': df_ttr["AND Gateway"].mean(),
        'All': df_ttr["All"].mean(),
    }
    new_df_ttr_mean_series = pd.Series(new_df_ttr_mean_dict)
    df_ttr_mean = df_ttr_mean.append(new_df_ttr_mean_series, ignore_index=True)

    # get mean ucer
    new_df_ucer_mean_dict = {
        'Actor': df_ucer["Actor"].mean(),
        'Activity': df_ucer["Activity"].mean(),
        'Activity Data': df_ucer["Activity Data"].mean(),
        'Further Specification': df_ucer["Further Specification"].mean(),
        'XOR Gateway': df_ucer["XOR Gateway"].mean(),
        'Condition Specification': df_ucer["Condition Specification"].mean(),
        'AND Gateway': df_ucer["AND Gateway"].mean(),
        'All': df_ucer["All"].mean(),
    }
    new_df_ucer_mean_series = pd.Series(new_df_ucer_mean_dict)
    df_ucer_mean = df_ucer_mean.append(new_df_ucer_mean_series, ignore_index=True)



    # get df complete series
    new_dict_series = {
        'F1 CRF': f_score_crf,
         'F1 Neural': f_score_neural,
         'F1 Relation': f_score_rel,
        'TTR': df_ttr_mean['All'][0],
        'UCER': df_ucer_mean['All'][0]}
    new_df_complete_series = pd.Series(data=new_dict_series)

    return new_df_complete_series, df_ttr, df_ucer, df_ttr_mean, df_ucer_mean

def evaluate_experiment_bert_filter(unaug_train_folds, aug_train_folds, f_score_crf, f_score_neural, f_score_rel):
    metrics: Metrics = Metrics(unaug_train_set=unaug_train_folds, train_set=aug_train_folds)
    df_ttr = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                   'Condition Specification', 'AND Gateway', 'All'])
    df_ttr_mean = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                        'Condition Specification', 'AND Gateway', 'All'])
    df_ucer = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                    'Condition Specification', 'AND Gateway', 'All'])
    df_ucer_mean = pd.DataFrame(columns=['Actor', 'Activity', 'Activity Data', 'Further Specification', 'XOR Gateway',
                                         'Condition Specification', 'AND Gateway', 'All'])
    df_bert = pd.DataFrame(columns=['Bert Score'])

    for i in range(5):
        # get TTR
        new_df_ttr_series = metrics.calculate_ttr(i)
        df_ttr = df_ttr.append(new_df_ttr_series, ignore_index=True)

        # get UCER
        new_df_ucer_series = metrics.calculate_ttr_trigram(i)
        df_ucer = df_ucer.append(new_df_ucer_series, ignore_index=True)

        # get Bert
        new_df_bert_series = metrics.calculate_bert_filter(i)
        df_bert = df_bert.append(new_df_bert_series, ignore_index=True)

    # get mean ttr
    new_df_ttr_mean_dict = {
        'Actor': df_ttr["Actor"].mean(),
        'Activity': df_ttr["Activity"].mean(),
        'Activity Data': df_ttr["Activity Data"].mean(),
        'Further Specification': df_ttr["Further Specification"].mean(),
        'XOR Gateway': df_ttr["XOR Gateway"].mean(),
        'Condition Specification': df_ttr["Condition Specification"].mean(),
        'AND Gateway': df_ttr["AND Gateway"].mean(),
        'All': df_ttr["All"].mean(),
    }
    new_df_ttr_mean_series = pd.Series(new_df_ttr_mean_dict)
    df_ttr_mean = df_ttr_mean.append(new_df_ttr_mean_series, ignore_index=True)

    # get mean ucer
    new_df_ucer_mean_dict = {
        'Actor': df_ucer["Actor"].mean(),
        'Activity': df_ucer["Activity"].mean(),
        'Activity Data': df_ucer["Activity Data"].mean(),
        'Further Specification': df_ucer["Further Specification"].mean(),
        'XOR Gateway': df_ucer["XOR Gateway"].mean(),
        'Condition Specification': df_ucer["Condition Specification"].mean(),
        'AND Gateway': df_ucer["AND Gateway"].mean(),
        'All': df_ucer["All"].mean(),
    }
    new_df_ucer_mean_series = pd.Series(new_df_ucer_mean_dict)
    df_ucer_mean = df_ucer_mean.append(new_df_ucer_mean_series, ignore_index=True)
    print(df_bert)
    # get mean bleu
    bert_mean = df_bert['Bert Score'].mean()

    # get df complete series
    new_dict_series = {
        'F1 CRF': f_score_crf,
         'F1 Neural': f_score_neural,
         'F1 Relation': f_score_rel,
        'TTR': df_ttr_mean['All'][0],
        'UCER': df_ucer_mean['All'][0],
        'BertScore': bert_mean}
    new_df_complete_series = pd.Series(data=new_dict_series)

    return new_df_complete_series, df_ttr, df_ucer, df_ttr_mean, df_ucer_mean, df_bert, bert_mean
