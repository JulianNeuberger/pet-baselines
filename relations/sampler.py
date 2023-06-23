import itertools
import math
import random

import typing

import data


def negative_sample(document: data.Document, num_positive: int, negative_rate: float,
                    verbose: bool = False) -> typing.List[typing.Tuple[int, int]]:
    num_negative_samples = math.ceil(negative_rate * num_positive)
    negative_samples = []

    candidates = list(itertools.combinations(range(len(document.mentions)), 2))
    candidates += [(t, h) for h, t in candidates]

    for head_mention_index, tail_mention_index in candidates:
        if len(negative_samples) >= num_negative_samples:
            break

        head_index = document.entity_index_for_mention(document.mentions[head_mention_index])
        tail_index = document.entity_index_for_mention(document.mentions[tail_mention_index])
        if document.relation_exists_between(head_index, tail_index):
            continue

        negative_samples.append((head_mention_index, tail_mention_index))

    if len(negative_samples) < num_negative_samples:
        if verbose:
            print(f'Could only build {len(negative_samples)}/{num_negative_samples} '
                  f'negative samples, as there were not enough candidates in {document.name}, '
                  f'reusing some.')
        missing_num_samples = num_negative_samples - len(negative_samples)
        while missing_num_samples > 0:
            negative_samples += negative_samples[:missing_num_samples]
            missing_num_samples = num_negative_samples - len(negative_samples)

        random.shuffle(negative_samples)

    return negative_samples
