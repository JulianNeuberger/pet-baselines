import copy
import typing
import random
from nltk.corpus import stopwords, wordnet
from numpy.random import shuffle
from random import random as rand
from augment import base, params
from data import model
from transformations import tokenmanager
import nltk

nltk.download("stopwords")


# Author: Benedikt
class Trafo100Step(base.AugmentationStep):
    def __init__(self, pos_type=True, prob=0.5, **kwargs):
        super().__init__(**kwargs)
        self.seed = 0
        self.prob = prob
        self.pos_type = pos_type
        self.stopwords = stopwords.words("english")

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.BooleanParameter(name="pos_type"),
            params.FloatParam(name="prob", min_value=0.0, max_value=1.0),
        ]

    def do_augment(self, doc2: model.Document):
        doc = copy.deepcopy(doc2)

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
                            syn for syn in synsets if syn.lower() != word.lower()
                        ]
                        # synsets = list(
                        #    set(synsets)
                        # )  # remove duplicate synonyms
                        if len(synsets) > 0 and random.random() < self.prob:
                            syn = synsets[0]
                            # syn = random.choice(synsets)
                            syn = syn.split("_")
                            for j in range(len(syn)):
                                token_to_insert = model.Token(
                                    text=syn[j],
                                    index_in_document=token.index_in_document + 1 + j,
                                    pos_tag=tokenmanager.get_pos_tag([syn[j]])[0],
                                    bio_tag=tokenmanager.get_bio_tag_based_on_left_token(
                                        token.bio_tag
                                    ),
                                    sentence_index=token.sentence_index,
                                )
                                tokenmanager.create_token(
                                    doc, token_to_insert, i + 1 + j
                                )
                                i += 1
                i += 1
        return doc
