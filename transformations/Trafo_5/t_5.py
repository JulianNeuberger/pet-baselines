import copy

from data import model
import typing
import data
from nltk.corpus import wordnet
import json

import numpy as np
docs: typing.List[model.Document] = data.loader.read_documents_from_json('../../complete.json')

def generate(max_outputs=1):



    for _ in range(max_outputs):

        for doc in docs:
            for sentence in doc.sentences:
                token_list = []
                counter = 0
                for token in sentence.tokens:
                    if token.pos_tag == "JJ":
                        wn_pos = "s"
                        antonyms = []
                        synsets = wordnet.synsets(token.text, wn_pos)
                        if synsets:
                            first_synset = synsets[0]
                            lemmas = first_synset.lemmas()
                            first_lemma = lemmas[0]
                            antonyms = first_lemma.antonyms()
                        if antonyms:
                            antonyms.sort(key=lambda x: str(x).split(".")[2])
                            token_2 = copy.deepcopy((token))
                            token_2.text = antonyms[0].name()
                            if token_2 not in token_list:
                                token_list.append(token_2)

                    #token_list = list(token_list)
                for token in token_list:
                    print(token.text)
                if len(token_list) % 2 == 0 and not is_ant_syn(token_list):
                    print(len(token_list))
                    for token_2 in token_list:
                        for token in sentence.tokens:

                            if token_2.index_in_document == token.index_in_document:
                                token.text = token_2.text


def is_ant_syn(token_list):
    for token_2 in token_list:
        for token_3 in token_list:
            if token_2 != token_3:
                if is_ant(token_2.text, token_3.text) and is_syn(token_2.text, token_3.text):
                    return True
    return False

def is_ant(word1, word2):
    antonyms = []
    for syn in wordnet.synsets(word1):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                antonyms.append(lemma.antonyms()[0].name())
    if word2 in antonyms:
        return True
    return False


def is_syn(word1, word2):
    synonyms = []
    for syn in wordnet.synsets(word1):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    if word2 in synonyms:
        return True
    return False

generate()
json_data = [doc.to_json_serializable() for doc in docs]

with open("./test_5.json", "w") as f:
    json.dump(json_data, f, indent=4)