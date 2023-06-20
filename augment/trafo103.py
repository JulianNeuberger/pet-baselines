from augment import base
from data import model
from nltk.corpus import wordnet
from transformations import tokenmanager
import copy
from random import choice
from random import random as rand

# Subsequence Substitution for Sequence Tagging


class Trafo103Step(base.AugmentationStep):

    def __init__(self, num_of_words = 2, prob: float = 0.5, kind_of_word = None):
        self.num_of_words = num_of_words
        self.kind_of_word = kind_of_word
        self.prob = prob

    def do_augment(self, doc: model.Document) -> model.Document:
        possible_sequences = []
        for i in range(len(doc.sentences)):
            num_of_words = self.num_of_words
            kind_of_words = copy.deepcopy(self.kind_of_word)
            sentence = doc.sentences[i]

            # length of the sentence must be greater than the number of words to be replaced (without punct)
            if len(sentence.tokens) - 1 >= num_of_words:
                # pos tags to be replaced
                pos_tags= []

                # to determine if the sentence has the (given) sequenz of pos-tags
                has_sequence_with_pos_tags = False

                # index in sentence of the first word to be replaced
                index_in_sentence = None

                # case1: no pos_tags given - determine pos_tags and optional index in sentence randomly
                if kind_of_words == None:

                    # choose randomly tokens with their pos_tags
                    tokens = copy.deepcopy(sentence.tokens)
                    first_token = choice(tokens)
                    pos_tags.append(first_token.pos_tag)
                    first_index_in_sentence = tokenmanager.get_index_in_sentence(sentence, [first_token.text], first_token.index_in_document)
                    index_in_sentence = first_index_in_sentence
                    for j in range(1, num_of_words):
                        try:
                            pos_tags.append(sentence.tokens[first_index_in_sentence + j].pos_tag)
                        except:
                            continue
                    kind_of_words = copy.deepcopy(pos_tags)
                    has_sequence_with_pos_tags = True
                else:
                    # case2: long enough sequence of pos-tags is given
                    # kind of words has to be the same length as the number of words to be changed

                    if len(kind_of_words) >= num_of_words:
                        while len(kind_of_words) > num_of_words:
                            kind_of_words.pop()

                        # search for such a sequence in the sentence
                        word_counter = 0
                        for k in range(len(sentence.tokens)-1):
                            if sentence.tokens[k].pos_tag == kind_of_words[word_counter]:
                                word_counter += 1
                                if word_counter == 1:
                                    index_in_sentence = k
                                if word_counter == len(kind_of_words):
                                    break
                            else:
                                word_counter = 0

                        # found such a sequence or not
                        if word_counter == num_of_words:
                            has_sequence_with_pos_tags = True
                        else:
                            has_sequence_with_pos_tags = False

                    # case3: less pos_tags are given: search for suitable pos_tags in the given sentence
                    elif len(kind_of_words) < num_of_words:
                        word_count = 0
                        for l in range(len(sentence.tokens)-1):
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
                            while len(kind_of_words) < num_of_words:
                                kind_of_words.append(sentence.tokens[index_in_sentence + word_count + j].pos_tag)
                                j += 1
                            has_sequence_with_pos_tags = True
                        else:
                            has_sequence_with_pos_tags = False
                    pos_tags = kind_of_words

                # only if a sequence of matching pos_tags was found
                if has_sequence_with_pos_tags and rand() <= self.prob:
                    #possible_sequences = []
                    # get all word-sequences from the document with the determined pos_tags
                    length = 0
                    if self.kind_of_word == None:
                        length = 0
                    else:
                        length = len(self.kind_of_word)
                    if (i == 0 and length >= num_of_words ) or self.kind_of_word == None or length < num_of_words:
                        possible_sequences = []
                        global word_c
                        word_c = 0
                        global seq
                        if kind_of_words != None:
                            for sent in doc.sentences:
                                word_c = 0
                                seq = []
                                for i in range(len(sent.tokens) - 1):
                                    if sent.tokens[i].pos_tag == kind_of_words[word_c]:
                                        word_c += 1
                                        seq.append(sent.tokens[i].text)
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
                        original_text = []
                        original_text.append(sentence.tokens[index_in_sentence].text)
                        for i in range(1, num_of_words):
                            original_text.append(sentence.tokens[index_in_sentence + i].text)
                        possible_seq = copy.deepcopy(possible_sequences)
                        possible_seq.remove(original_text)

                        # only if there are other possible sequences the original text can be changed
                        if len(possible_seq) > 0:
                            # choose the new text out of the possibilities
                            new_text = choice(possible_seq)
                            for m in range(num_of_words):
                                sentence.tokens[index_in_sentence + m].text = new_text[m]
        return doc