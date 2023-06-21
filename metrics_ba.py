import copy
import random
import typing
from scipy.stats import gmean
import pandas as pd
import bert_score
from data import model
import torch
from nltk.translate.bleu_score import sentence_bleu


class Metrics:
    def __init__(self, train_set: typing.List[model.Document], unaug_train_set: typing.List[model.Document]):
        self.train_set = train_set
        self.unaug_train_set = unaug_train_set
        self.entity_list = ["Actor", "Activity", "Activity Data", "Further Specification", "XOR Gateway",
                            "Condition Specification", "AND Gateway"]

    def calculate_ttr(self, fold_number):
        ttr_list = {}
        for ner in self.entity_list:
            entity_counter = 0
            all_counter = 0
            unique_token = []
            unique_token_all = []
            entity_ttr = 0
            for doc in self.train_set[fold_number]:
                for sentence in doc.sentences:
                    for token in sentence.tokens:
                        if token.bio_tag != "O":
                            bio_list = token.bio_tag.split("-")
                            if len(bio_list) > 1:
                                all_counter += 1
                                bio = bio_list[1]
                                lower_text_all = token.text.lower()
                                if lower_text_all not in unique_token_all:
                                    unique_token_all.append(lower_text_all)
                                if bio == ner:
                                    entity_counter += 1
                                    lower_text = token.text.lower()
                                    if lower_text not in unique_token:
                                        unique_token.append(lower_text)
            if entity_counter != 0:
                entity_ttr = len(unique_token) / entity_counter
                ttr_list[ner] = entity_ttr
            else:
                ttr_list[ner] = 0
            ttr_list["All"] = len(unique_token_all) / all_counter
        # print(ttr_list)
        new_series = pd.Series(data=ttr_list)
        return new_series

    def calculate_ttr_trigram(self, fold_number):
        ttr_list = {}
        for ner in self.entity_list:
            unique_trigrams_list = []
            unique_trigrams_list_all = []
            entity_counter = 0
            all_counter = 0
            for doc in self.train_set[fold_number]:
                for sentence in doc.sentences:
                    for i in range(len(sentence.tokens)):
                        if 0 < i < len(sentence.tokens):
                            if sentence.tokens[i].bio_tag != "O":
                                bio_list = sentence.tokens[i].bio_tag.split("-")
                                if len(bio_list) > 1:
                                    bio = bio_list[1]
                                    all_counter += 1
                                    trigram = [sentence.tokens[i - 1].text.lower(), sentence.tokens[i].text.lower(),
                                               sentence.tokens[i + 1].text.lower()]
                                    if trigram not in unique_trigrams_list_all:
                                        unique_trigrams_list_all.append(trigram)
                                    if bio == ner:
                                        entity_counter += 1
                                        if trigram not in unique_trigrams_list:
                                            unique_trigrams_list.append(trigram)
            if entity_counter != 0:
                entity_ttr = len(unique_trigrams_list) / entity_counter
                ttr_list[ner] = entity_ttr
            else:
                ttr_list[ner] = 0
            ttr_list["All"] = len(unique_trigrams_list_all) / all_counter
        new_series = pd.Series(data=ttr_list)
        return new_series

    def calculate_bert_old(self, fold_number):
        predictions = []
        references = []
        for doc in self.train_set[fold_number]:
            curr_sentence = ""
            for sentence in doc.sentences:
                for token in sentence.tokens:
                    curr_sentence += " "
                    curr_sentence += token.text
                predictions.append(curr_sentence)
        for doc in self.unaug_train_set[fold_number]:
            curr_sentence = ""
            for sentence in doc.sentences:
                for token in sentence.tokens:
                    curr_sentence += " "
                    curr_sentence += token.text
                references.append(curr_sentence)
        results = bert_score.score(predictions, references, lang="en", verbose=True)
        bert_list = {"Bert Score": results[2]}
        new_series = pd.Series(data=bert_list)
        return new_series


    def calculate_bert(self, fold_number):
        predictions = []
        references = []
        for doc in self.train_set[fold_number]:
            curr_sentence = ""
            for sentence in doc.sentences:
                for token in sentence.tokens:
                    curr_sentence += " "
                    curr_sentence += token.text
                predictions.append(curr_sentence)
        for doc in self.unaug_train_set[fold_number]:
            curr_sentence = ""
            for sentence in doc.sentences:
                for token in sentence.tokens:
                    curr_sentence += " "
                    curr_sentence += token.text
                references.append(curr_sentence)
        results = bert_score.score(predictions, references, lang="en", verbose=True)
        mean = torch.mean(results[2])
        mean_as_float = float(mean)
        bert_list = {"Bert Score": mean_as_float}
        new_series = pd.Series(data=bert_list)
        return new_series


    def calculate_bleu(self, fold_number):
        tup = []
        # per document
        for j in range(len(self.train_set[fold_number])):
            index_unaug = 0
            if j < 0.5* len(self.train_set[fold_number]):
                index_unaug = j
            else:
                index_unaug = int(j - (0.5* len(self.train_set[fold_number])))

            # per sentence in document
            for i in range(len(self.train_set[fold_number][j].sentences)):
                sentence1 = []
                for token in self.train_set[fold_number][j].sentences[i].tokens:
                    sentence1.append(token.text)
                sentence2 = []
                for token in self.unaug_train_set[fold_number][index_unaug].sentences[i].tokens:
                    sentence2.append(token.text)
                tup.append((sentence1, sentence2))
        trigram = []
        for tu in tup:
            trigram.append(sentence_bleu([tu[0]], tu[1], weights=(1, 0, 1, 0)))
        bleu_list = {"Bleu Score": gmean(trigram)}
        new_series = pd.Series(data=bleu_list)
        #return gmean(trigram)
        return new_series

tokens = [model.Token(text="I", index_in_document=0,
                      pos_tag="PRP",
                      bio_tag="B-Actor",
                      sentence_index=0),
          model.Token(text="leave", index_in_document=1,
                      pos_tag="VBP",
                      bio_tag="O",
                      sentence_index=0),
          model.Token(text="Human", index_in_document=2,
                      pos_tag="NN",
                      bio_tag="B-Activity Data",
                      sentence_index=0),
          model.Token(text="Human", index_in_document=3,
                      pos_tag="NNS",
                      bio_tag="I-Activity Data",
                      sentence_index=0),
          model.Token(text="leave", index_in_document=4,
                      pos_tag="VBP",
                      bio_tag="O",
                      sentence_index=0),
          model.Token(text="Human", index_in_document=5,
                      pos_tag="NN",
                      bio_tag="B-Activity Data",
                      sentence_index=0),
          model.Token(text="Human", index_in_document=6,
                      pos_tag="NNS",
                      bio_tag="I-Activity Data",
                      sentence_index=0),
          model.Token(text="Human", index_in_document=7,
                      pos_tag="JJ",
                      bio_tag="O",
                      sentence_index=0),
          model.Token(text=".", index_in_document=8,
                      pos_tag=".",
                      bio_tag=".",
                      sentence_index=0)
          ]

sentence1 = model.Sentence(tokens=tokens)

doc = model.Document(
    text="I leave HR office",
    name="1", sentences=[sentence1],
    mentions=[],
    entities=[],
    relations=[])
train_set = [doc]
tokens2 = [model.Token(text="I", index_in_document=0,
                      pos_tag="PRP",
                      bio_tag="B-Actor",
                      sentence_index=0),
          model.Token(text="leave", index_in_document=1,
                      pos_tag="VBP",
                      bio_tag="O",
                      sentence_index=0),
          model.Token(text="car", index_in_document=2,
                      pos_tag="NN",
                      bio_tag="B-Activity Data",
                      sentence_index=0),
          model.Token(text="Human", index_in_document=3,
                      pos_tag="NNS",
                      bio_tag="I-Activity Data",
                      sentence_index=0),
          model.Token(text="leave", index_in_document=4,
                      pos_tag="VBP",
                      bio_tag="O",
                      sentence_index=0),
          model.Token(text="Human", index_in_document=5,
                      pos_tag="NN",
                      bio_tag="B-Activity Data",
                      sentence_index=0),
          model.Token(text="Human", index_in_document=6,
                      pos_tag="NNS",
                      bio_tag="I-Activity Data",
                      sentence_index=0),
          model.Token(text=".", index_in_document=7,
                      pos_tag=".",
                      bio_tag=".",
                      sentence_index=0)
          ]

sentence2 = model.Sentence(tokens=tokens2)
tokens3 = copy.deepcopy(tokens2)
sentence3 = model.Sentence(tokens=tokens3)
sentence3.tokens[2].text = "face"
doc2 = model.Document(
    text="I leave HR office",
    name="1", sentences=[sentence2],
    mentions=[],
    entities=[],
    relations=[])
doc3 = model.Document(
    text="I leave HR office",
    name="1", sentences=[sentence3],
    mentions=[],
    entities=[],
    relations=[])
train_set2 = [doc2, doc2]
train_set3 = [doc3, doc3]
met = Metrics(train_set=train_set3, unaug_train_set=train_set2)



#print(ergebnis)