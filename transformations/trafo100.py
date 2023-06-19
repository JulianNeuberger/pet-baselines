import random
from transformations import tokenmanager
import nltk
import spacy
from nltk.corpus import stopwords, wordnet

from data import model


class Trafo100:

    def __init__(self, seed=0, prob=1):
        self.seed = seed
        self.prob = prob
        self.stopwords = stopwords.words("english")

    def transform(self, doc: model.Document):
        random.seed(self.seed)
        pos_wordnet_dict = {
            "VB": "v",
            "VBD": "v",
            "VBG": "v",
            "VBN": "v",
            "VBP": "v",
            "VBZ": "v",
            "NN": "n",
            "NNS": "n",
            "NNP": "n",
            "NNPS": "n",
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
                        #synsets = list(
                        #    set(synsets)
                       # )  # remove duplicate synonyms
                        if len(synsets) > 0 and random.random() < self.prob:
                            syn = synsets[0]
                            #syn = random.choice(synsets)
                            syn = syn.split("_")
                            for j in range(len(syn)):

                                token_to_insert = model.Token(text=syn[j], index_in_document=token.index_in_document + 1 + j,
                                                              pos_tag=tokenmanager.get_pos_tag([syn[j]])[0],
                                                              bio_tag=tokenmanager.get_bio_tag_based_on_left_token(
                                                                  token.bio_tag),
                                                              sentence_index=token.sentence_index)
                                tokenmanager.create_token(doc, token_to_insert, i + 1 + j)
                                i += 1
                i += 1
        return doc

tokens = [model.Token(text="good", index_in_document=0,
                          pos_tag="JJ", bio_tag="O",
                          sentence_index=0), model.Token(text="leave", index_in_document=1,
                                                         pos_tag="VBP", bio_tag="O",
                                                         sentence_index=0), model.Token(text="head", index_in_document=2,
                                                                                        pos_tag="NN", bio_tag="O",
                                                                                        sentence_index=0), model.Token(text=".", index_in_document=3,
                                                                                        pos_tag=".", bio_tag="O",
                                                                                        sentence_index=0)]

sentence1 = model.Sentence(tokens=tokens)

doc = model.Document(
    text="I leave HR office",
    name="1", sentences=[sentence1],
    mentions=[],
    entities=[],
    relations=[])

trafo = Trafo100()
print(trafo.transform(doc))