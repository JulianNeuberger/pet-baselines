import copy
import random
import typing
from random import choice
from random import random as rand

from augment import base, params
from data import model
from transformations import tokenmanager


# Subsequence Substitution for Sequence Tagging


# Author: Leonie
class Trafo103Step(base.AugmentationStep):
    def __init__(self, num_of_words=2, prob: float = 0.5, pos_tags=None, **kwargs):
        super().__init__(**kwargs)
        self.num_of_words = num_of_words
        self.pos_tags = pos_tags
        self.prob = prob

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        possible_pos_tags = [
            "JJ",
            "JJS",
            "JJR",
            "NN",
            "NNS",
            "NNP",
            "NNPS",
            "RB",
            "RBS",
            "RBR",
            "DT",
            "IN",
            "VBN",
            "VBP",
            "VBZ",
            "PRP",
            "WP",
        ]
        return [
            params.FloatParam(name="prob", min_value=0, max_value=1),
            params.IntegerParam(name="num_of_words", min_value=0, max_value=5),
            params.ChoiceParam(
                name="pos_tags",
                choices=possible_pos_tags,
                max_num_picks=len(possible_pos_tags),
            ),
        ]

    def do_augment(self, doc2: model.Document) -> model.Document:
        doc = copy.deepcopy(doc2)
        possible_sequences = []
        for i in range(len(doc.sentences)):
            kind_of_words = copy.deepcopy(self.pos_tags)
            sentence = doc.sentences[i]

            # length of the sentence must be greater than the number of words to be replaced (without punct)
            if len(sentence.tokens) - 1 < self.num_of_words:
                continue

            # to determine if the sentence has the (given) sequence of pos-tags
            has_sequence_with_pos_tags = False

            # index in sentence of the first word to be replaced
            index_in_sentence = None

            # case1: no pos_tags given - determine pos_tags and optional index in sentence randomly
            if kind_of_words is None:
                # choose random tokens with their pos_tags
                tokens = copy.deepcopy(sentence.tokens)
                tokens = tokens[: -self.num_of_words]

                first_token = random.choice(tokens)
                first_index_in_sentence = tokenmanager.get_index_in_sentence(
                    sentence, [first_token.text], first_token.index_in_document
                )
                last_index_in_sentence = first_index_in_sentence + self.num_of_words + 1
                token_sequence = sentence.tokens[
                    first_index_in_sentence:last_index_in_sentence
                ]
                pos_tags = [t.pos_tag for t in token_sequence]

                index_in_sentence = first_index_in_sentence

                kind_of_words = copy.deepcopy(pos_tags)
                has_sequence_with_pos_tags = True
            else:
                # case2: long enough sequence of pos-tags is given
                # kind of words has to be the same length as the number of words to be changed
                if len(kind_of_words) >= self.num_of_words:
                    while len(kind_of_words) > self.num_of_words:
                        kind_of_words.pop()

                    # search for such a sequence in the sentence
                    word_counter = 0
                    for k in range(len(sentence.tokens) - 1):
                        if sentence.tokens[k].pos_tag == kind_of_words[word_counter]:
                            word_counter += 1
                            if word_counter == 1:
                                index_in_sentence = k
                            if word_counter == len(kind_of_words):
                                break
                        else:
                            word_counter = 0

                    # found such a sequence or not
                    has_sequence_with_pos_tags = word_counter == self.num_of_words

                # case3: less pos_tags are given: search for suitable pos_tags in the given sentence
                elif len(kind_of_words) < self.num_of_words:
                    word_count = 0
                    for l in range(len(sentence.tokens) - 1):
                        if sentence.tokens[l].pos_tag == kind_of_words[word_count]:
                            word_count += 1
                            if word_count == 1:
                                index_in_sentence = l
                            if word_count == len(kind_of_words):
                                break
                        else:
                            word_count = 0

                    # found matching sequence of pos_tags in the sentence
                    if word_count == len(kind_of_words):
                        j = 0
                        while len(kind_of_words) < self.num_of_words:
                            kind_of_words.append(
                                sentence.tokens[
                                    index_in_sentence + word_count + j
                                ].pos_tag
                            )
                            j += 1
                        has_sequence_with_pos_tags = True
                    else:
                        has_sequence_with_pos_tags = False

            # only if a sequence of matching pos_tags was found
            if has_sequence_with_pos_tags and rand() <= self.prob:
                # possible_sequences = []
                # get all word-sequences from the document with the determined pos_tags
                length = 0
                if self.pos_tags is None:
                    length = 0
                else:
                    length = len(self.pos_tags)
                if (
                    (i == 0 and length >= self.num_of_words)
                    or self.pos_tags is None
                    or length < self.num_of_words
                ):
                    possible_sequences = []
                    if kind_of_words is not None:
                        for sent in doc.sentences:
                            word_c = 0
                            seq = []
                            for m in range(len(sent.tokens) - 1):
                                if sent.tokens[m].pos_tag == kind_of_words[word_c]:
                                    word_c += 1
                                    seq.append(sent.tokens[m].text)
                                    if word_c == len(kind_of_words):
                                        possible_sequences.append(seq)
                                        seq = []
                                        word_c = 0
                                else:
                                    word_c = 0
                                    seq = []
                else:
                    pass
                if len(possible_sequences) >= 1:
                    # get the original text to delete it from possible_sequences
                    original_text = [sentence.tokens[index_in_sentence].text]
                    for j in range(1, self.num_of_words):
                        original_text.append(
                            sentence.tokens[index_in_sentence + j].text
                        )
                    possible_seq = copy.deepcopy(possible_sequences)

                    try:
                        possible_seq.remove(original_text)
                    except:
                        pass
                    # only if there are other possible sequences the original text can be changed
                    if len(possible_seq) > 0:
                        # choose the new text out of the possibilities
                        new_text = choice(possible_seq)
                        for m in range(num_of_words):
                            sentence.tokens[index_in_sentence + m].text = new_text[m]
        return doc
