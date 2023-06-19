import copy
from random import random

from augment import base
from nltk.corpus import wordnet
from data import model


class Trafo5Step(base.AugmentationStep):
    def __init__(self, p):
        self.p = p
    def do_augment(self, doc: model.Document) -> model.Document:
        doc = doc.copy()
        # Choose from Adjectives and Adverbs
        pos_tags = ["JJ", "JJS", "JJR", "RB", "RBR", "RBS"]
        # for all sentences and all tokens in the Document
        for sentence in doc.sentences:
            token_list = []
            for token in sentence.tokens:
                # when the text of the token has the wanted POS Tag go on
                if token.pos_tag in pos_tags:
                    wn_pos = "s"
                    antonyms = []
                    synsets = wordnet.synsets(token.text, wn_pos)
                    # get the antonym of the word and put it in a token list with the new Tokens
                    if synsets:
                        first_synset = synsets[0]
                        lemmas = first_synset.lemmas()
                        first_lemma = lemmas[0]
                        antonyms = first_lemma.antonyms()
                    if antonyms:
                        antonyms.sort(key=lambda x: str(x).split(".")[2])
                        token_2 = copy.deepcopy(token)
                        token_2.text = antonyms[0].name()
                        if token_2 not in token_list:
                            token_list.append(token_2)
            # if there is even number of adjectives or adverbs and if list are synonyms or antonyms --> replace tokens
            if len(token_list) % 2 == 0 and not Trafo5Step.is_ant_syn(self, token_list) and random() < self.p:
                for token_2 in token_list:
                    for token in sentence.tokens:
                        if token_2.index_in_document == token.index_in_document:
                            token.text = token_2.text
        return doc

    # returns, whether words in a token list are synonyms or antonyms
    def is_ant_syn(self, token_list):
        for token_2 in token_list:
            for token_3 in token_list:
                if token_2 != token_3:
                    if Trafo5Step.is_ant(self, token_2.text, token_3.text) \
                            or Trafo5Step.is_syn(self, token_2.text, token_3.text):
                        return True
        return False

    # returns, whether two words are antonyms
    def is_ant(self, word1, word2):
        antonyms = []
        for syn in wordnet.synsets(word1):
            for lemma in syn.lemmas():
                if lemma.antonyms():
                    antonyms.append(lemma.antonyms()[0].name())
        if word2 in antonyms:
            return True
        return False

    # returns, whether two words are synonyms
    def is_syn(self, word1, word2):
        synonyms = []
        for syn in wordnet.synsets(word1):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        if word2 in synonyms:
            return True
        return False
