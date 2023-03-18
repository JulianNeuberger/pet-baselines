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


def entity_f1_stats(*, predicted_documents: typing.List[data.Document],
                    ground_truth_documents: typing.List[data.Document],
                    min_num_mentions: int = 1,
                    verbose: bool = False) -> typing.Tuple[float, float, float]:
    predicted_documents = [d.copy() for d in predicted_documents]
    for d in predicted_documents:
        d.entities = [e for e in d.entities if len(e.mention_indices) >= min_num_mentions]

    ground_truth_documents = [d.copy() for d in ground_truth_documents]
    for d in ground_truth_documents:
        d.entities = [e for e in d.entities if len(e.mention_indices) >= min_num_mentions]

    return _f1_stats(predicted_documents=predicted_documents,
                     ground_truth_documents=ground_truth_documents,
                     attribute='entities', verbose=verbose)


def _f1_stats(*, predicted_documents: typing.List[data.Document],
              ground_truth_documents: typing.List[data.Document],
              attribute: str,
              verbose: bool = False):
    assert attribute in ['mentions', 'relations', 'entities']
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
        assert len(pred_as_set) == len(pred_attribute), f'{pred_as_set}, {pred_attribute}'

        ok_preds = true_as_set.intersection(pred_as_set)
        non_ok = [e.pretty_print(p) for e in pred_attribute if e.to_tuple(p) not in true_as_set]
        if verbose and len(non_ok) > 0:
            print('=' * 150)
            print(p.text)
            print('pred mentions:')
            print(', '.join([e.pretty_print(p) for e in p.mentions]))
            print()
            print('true mentions')
            print(', '.join([e.pretty_print(t) for e in t.mentions]))
            print('-' * 100)
            print('pred')
            print(', '.join([a.pretty_print(p) for a in pred_attribute]))
            print([e for e in pred_as_set])
            print('-' * 100)
            print('true')
            print(', '.join([a.pretty_print(t) for a in true_attribute]))
            print([e for e in true_as_set])
            print('-' * 100)
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
