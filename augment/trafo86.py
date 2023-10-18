from augment import base
from data import model
from nltk.corpus import wordnet
from transformations import tokenmanager
import copy
from random import random as rand
from random import shuffle

# Replace nouns with hyponyms or hypernyms - Wortebene

# Author: Leonie
class Trafo86Step(base.AugmentationStep):
    def __init__(self, max_noun: int = 1, kind_of_replace: int = 2, no_dupl: bool = False, prob:float = 0.5):
        self.max_noun = max_noun
        self.kind_of_replace = kind_of_replace
        self.no_dupl = no_dupl # if True: no duplicates
        self.prob = prob

    def do_augment(self, doc: model.Document) -> model.Document:
        doc = doc.copy()
        for sentence in doc.sentences:

            # create list with all tokens with pos_tag noun and a list with their indices in the sentence
            tok_list = []
            dupl_list = []
            index_in_sentence_list = []
            has_hyp = False
            counter = 0
            change_pos = 0
            for token in sentence.tokens:
                if token.pos_tag in ["NN", "NNS", "NNP", "NNPS"]:
                    if self.no_dupl:
                        if not token.text in dupl_list:
                            tok_list.append(copy.deepcopy(token))

                            index_in_sentence_list.append(counter)
                            dupl_list.append(copy.deepcopy(token.text))
                    else:
                        tok_list.append(copy.deepcopy(token))
                        index_in_sentence_list.append(counter)
                        dupl_list.append(copy.deepcopy(token.text))
                counter += 1

            # sentence must contain a noun
            if len(tok_list) >= 1:

                # token_list will be shuffled, tok_list contains the right order of tokens
                token_list = copy.deepcopy(tok_list)

                # shuffle noun-list for random noun selection
                shuffle(token_list)
                print("=================")
                print(tok_list)
                print(token_list)
                print(token.text)
                # if more than the actual number of nouns should be replaced, set maxi_noun to the actual number of nouns
                maxi_noun = self.max_noun
                if maxi_noun > len(tok_list):
                    maxi_noun = len(tok_list)

                # change only the maximum amount of nouns
                noun_count = 0
                while noun_count < maxi_noun:
                    if rand() <= self.prob:
                        kind_of_replacement = self.kind_of_replace

                        # noun to be changed
                        token = token_list[noun_count]

                        # search for the index in sentence
                        index_in_sentence = None
                        print("------------------")
                        print(tok_list)
                        print(token_list)
                        print(token.text)
                        for i in range(0, len(tok_list)):
                            if token.text == tok_list[i].text:
                                index_in_sentence = index_in_sentence_list[i]
                                break

                        # determine the kind of replacement
                        if kind_of_replacement == 2:
                            num = rand()
                            if num <= 0.5:
                                kind_of_replacement = 0
                            else:
                                kind_of_replacement = 1

                        # replace with a hypernym
                        if kind_of_replacement == 1:
                            hypernyms = []
                            synsets = wordnet.synsets(token.text, "n")
                            if synsets:
                                syn = synsets[0]
                                hypernyms = wordnet.synset(syn.name()).hypernyms()
                            if hypernyms:
                                hyp = hypernyms[0]
                                hype = hyp.name()
                                hypern = hype.split(".", 1)
                                token.text = hypern[0]
                                has_hyp = True

                        # replace with a hyponym
                        elif kind_of_replacement == 0:  # hyponym
                            hyponyms = []
                            synsets = wordnet.synsets(token.text, "n")
                            if synsets:
                                syn = synsets[0]
                                hyponyms = wordnet.synset(syn.name()).hyponyms()
                            if hyponyms:
                                hyp = hyponyms[0]
                                hypo = hyp.name()
                                hypon = hypo.split(".", 1)
                                token.text = hypon[0]
                                has_hyp = True

                        # only if a hypernym/ hyponym exists
                        if has_hyp:

                            # split the token.text if it contains several words
                            text = token.text.split("_")

                            # set the text and pos_tag of the first token
                            token.text = text[0]
                            token.pos_tag = tokenmanager.get_pos_tag([text[0]])[0]
                            print(index_in_sentence)
                            print(change_pos)
                            print("----------------")
                            # set the first token
                            sentence.tokens[index_in_sentence + change_pos].text = token.text
                            sentence.tokens[index_in_sentence + change_pos].pos_tag = token.pos_tag

                            # if the hypernym/hyponym has several words, for each further word create a new token
                            if len(text) > 1:

                                # generate bio-tag
                                bio_tag = tokenmanager.get_bio_tag_based_on_left_token(token.bio_tag)

                                # get mention index
                                ment_ind = tokenmanager.get_mentions(doc, index_in_sentence + change_pos, token.sentence_index)
                                # create Tokens
                                for i in range(1, len(text)):
                                    tok = model.Token(text=text[i], index_in_document=token.index_in_document + i + change_pos,
                                                      pos_tag=tokenmanager.get_pos_tag([text[i]])[0], bio_tag=bio_tag,
                                                      sentence_index=token.sentence_index)
                                    if ment_ind == []:
                                        tokenmanager.create_token(doc, tok, index_in_sentence + i + change_pos, None)
                                    else:
                                        tokenmanager.create_token(doc, tok, index_in_sentence + i + change_pos, ment_ind[0])
                                change_pos += len(text) - 1
                    noun_count += 1
        return doc
