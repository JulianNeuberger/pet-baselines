import copy
from random import random

from transformers import pipeline

from data import model
from augment import base
from transformations import tokenmanager


class Trafo58Step(base.AugmentationStep):

    def __init__(self, p: float = 1, lang: str = "de"):
        self.lang = lang
        self.p = p

    def do_augment(self, doc: model.Document):
        for sentence in doc.sentences:
            i = 0
            while i < len(sentence.tokens) - 1:
                token = sentence.tokens[i]
                current_bio = tokenmanager.get_bio_tag_short(token.bio_tag)
                text_before = ""
                j = 0
                while tokenmanager.get_bio_tag_short(sentence.tokens[i + j].bio_tag) == current_bio:
                    if i + j < len(sentence.tokens) - 1:
                        if j == 0:
                            text_before += sentence.tokens[i + j].text
                        else:
                            text_before += " "
                            text_before += sentence.tokens[i + j].text
                    j += 1
                    if i + j > len(sentence.tokens) - 1:
                        break

                if random() >= self.p:
                    i += j
                    continue
                if text_before in [",", ".", "?", "!", ":", "#", "-"]:
                    translated = text_before
                else:
                    translated = self.back_translate(text_before)
                text_before_list = text_before.split()
                translated_list = translated.split()
                diff = len(translated_list) - len(text_before_list)
                if diff > 0:
                    token.text = translated_list[0]
                    token.pos_tag = tokenmanager.get_pos_tag([token.text])[0]
                    if len(translated_list) > 1:
                        for k in range(1, len(translated_list)):
                            tok = model.Token(text=translated_list[k],
                                              index_in_document=token.index_in_document + i + k,
                                              pos_tag=tokenmanager.get_pos_tag([token.text])[0],
                                              bio_tag=tokenmanager.get_bio_tag_based_on_left_token(token.bio_tag),
                                              sentence_index=token.sentence_index)
                            tokenmanager.create_token(doc, tok, i + k)
                elif diff == 0:
                    for k in range(0, len(translated_list)):
                        index_in_doc = token.index_in_document
                        sentence.tokens[i + k].text = translated_list[k]
                        sentence.tokens[i + k].pos_tag = tokenmanager.get_pos_tag([token.text])[0]
                else:
                    for k in range(len(translated_list)):
                        sentence.tokens[i + k].text = translated_list[k]
                        sentence.tokens[i + k].pos_tag = tokenmanager.get_pos_tag([token.text])[0]
                    for k in range(len(translated_list), len(text_before_list)):
                        tokenmanager.delete_token(doc, sentence.tokens[i + len(translated_list)].index_in_document)
                i = i + j + diff
        return doc

    #  returns the back translated text, when it's not working, it returns the old text
    def back_translate(self, en: str):
        try:
            en_new = self.encode_decode(en)
        except Exception as ex:
            print(ex)
            print("Returning Default due to Run Time Exception")
            en_new = en
        return en_new

    def encode_decode(self, text):
        # translate and un-translate
        # using Helsinki-NLP OpusMT models
        encode = pipeline(
            "translation_en_to_{}".format(self.lang),
            model="Helsinki-NLP/opus-mt-en-{}".format(self.lang),
            device=-1,
        )
        decode = pipeline(
            "translation_{}_to_en".format(self.lang),
            model="Helsinki-NLP/opus-mt-{}-en".format(self.lang),
            device=-1,
        )
        # en->lang->en
        return decode(
            encode(text, max_length=600)[0]["translation_text"],
            max_length=600,
        )[0]["translation_text"]
