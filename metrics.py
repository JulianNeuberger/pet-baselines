import collections
import typing

import data


def relation_f1_stats(*, predicted_documents: typing.List[data.Document],
                      ground_truth_documents: typing.List[data.Document],
                      verbose: bool = False) -> typing.Tuple[float, float, float]:
    return _f1_stats(predicted_documents=predicted_documents,
                     ground_truth_documents=ground_truth_documents,
                     attribute='relations', verbose=verbose)


def mentions_f1_stats(*, predicted_documents: typing.List[data.Document],
                      ground_truth_documents: typing.List[data.Document],
                      verbose: bool = False) -> typing.Tuple[float, float, float]:
    return _f1_stats(predicted_documents=predicted_documents,
                     ground_truth_documents=ground_truth_documents,
                     attribute='mentions', verbose=verbose)


def _f1_stats(*, predicted_documents: typing.List[data.Document],
              ground_truth_documents: typing.List[data.Document],
              attribute: str,
              verbose: bool = False):
    assert attribute in ['mentions', 'relations']
    assert len(predicted_documents) == len(ground_truth_documents)

    num_gold = 0
    num_pred = 0
    num_ok = 0
    for p, t in zip(predicted_documents, ground_truth_documents):
        true_attribute = getattr(t, attribute)
        pred_attribute = getattr(p, attribute)

        num_gold += len(true_attribute)
        num_pred += len(pred_attribute)

        true_as_set = set([e.to_tuple(t) for e in true_attribute])
        assert len(true_as_set) == len(true_attribute), f'{len(true_as_set)}, {len(true_attribute)}'

        pred_as_set = set([e.to_tuple(p) for e in pred_attribute])
        assert len(pred_as_set) == len(pred_attribute), f'{len(pred_as_set)}, {len(pred_attribute)}'

        ok_preds = true_as_set.intersection(pred_as_set)
        non_ok = [e.pretty_print(p) for e in pred_attribute if e.to_tuple(p) not in true_as_set]
        if attribute == 'relations' and verbose and len(non_ok) > 0:
            print('='*150)
            print(p.text)
            print('pred mentions:')
            print(', '.join([e.pretty_print(p) for e in p.mentions if 'act' in e.ner_tag.lower()]))
            print()
            print('true mentions')
            print(', '.join([e.pretty_print(t) for e in t.mentions if 'act' in e.ner_tag.lower()]))
            print('-'*100)
            print('pred')
            print(', '.join([a.pretty_print(p) for a in pred_attribute if 'act' in a.tag.lower()]))
            print([e for e in pred_as_set if 'act' in e[1]])
            print('-'*100)
            print('true')
            print(', '.join([a.pretty_print(t) for a in true_attribute if 'act' in a.tag.lower()]))
            print([e for e in true_as_set if 'act' in e[1]])
            print('-'*100)
            print('ok')
            print(ok_preds)
            print('non ok')
            print(non_ok)
        num_ok += len(ok_preds)

    if num_pred == 0 and num_gold == 0:
        precision = 1.0
    elif num_pred == 0 and num_gold != 0:
        precision = 0.0
    else:
        precision = num_ok / num_pred

    if num_gold == 0 and num_pred == 0:
        recall = 1.0
    elif num_gold == 0 and num_pred != 0:
        recall = 0.0
    else:
        recall = num_ok / num_gold if num_gold != 0 else 0

    if verbose:
        print(f'{num_gold}, {num_pred}, {num_ok}')

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0.0 else 0.0

    return precision, recall, f1
