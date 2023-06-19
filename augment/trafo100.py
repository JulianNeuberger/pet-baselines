import copy
import typing
import random
from nltk.corpus import stopwords, wordnet
from numpy.random import shuffle
from random import random as rand
from augment import base
from data import model
from transformations import tokenmanager

class Trafo100Step(base.AugmentationStep):
    def __init__(self, pos_type, prob=1):
        self.seed = 0
        self.prob = prob
        self.pos_type = pos_type
        self.stopwords = stopwords.words("english")

    def do_augment(self,  doc: model.Document):
        random.seed(self.seed)
        if self.pos_type:
            pos_wordnet_dict = {
                "NN": "n",
                "NNS": "n",
                "NNP": "n",
                "NNPS": "n",
            }
        else:
            pos_wordnet_dict = {
                "RB": "r",
                "RBR": "r",
                "RBS": "r",
                "JJ": "s",
                "JJR": "s",
                "JJS": "s",
        }
        for sentence in doc.sentences:
            i = 0
            while i < len(sentence.tokens):
                token = sentence.tokens[i]
                word = token.text
                wordnet_pos = pos_wordnet_dict.get(token.pos_tag)
                if not wordnet_pos:
                    pass
                elif word in self.stopwords:
                    pass
                else:
                    synsets = wordnet.synsets(word, pos=wordnet_pos)
                    if len(synsets) > 0:
                        synsets = [syn.name().split(".")[0] for syn in synsets]
                        synsets = [
                            syn
                            for syn in synsets
                            if syn.lower() != word.lower()
                        ]
                        # synsets = list(
                        #    set(synsets)
                        # )  # remove duplicate synonyms
                        if len(synsets) > 0 and random.random() < self.prob:
                            syn = synsets[0]
                            # syn = random.choice(synsets)
                            syn = syn.split("_")
                            for j in range(len(syn)):
                                token_to_insert = model.Token(text=syn[j],
                                                              index_in_document=token.index_in_document + 1 + j,
                                                              pos_tag=tokenmanager.get_pos_tag([syn[j]])[0],
                                                              bio_tag=tokenmanager.get_bio_tag_based_on_left_token(
                                                                  token.bio_tag),
                                                              sentence_index=token.sentence_index)
                                tokenmanager.create_token(doc, token_to_insert, i + 1 + j)
                                i += 1
                i += 1
        return doc