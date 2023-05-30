import copy

from transformers import pipeline

from data import model
from augment import base
from transformations import tokenmanager


class Trafo54Step(base.AugmentationStep):

    def __init__(self, adj_adv: bool, nn: bool, vb: bool, lang: str = "de"):
        self.adj_adv = adj_adv
        self.nn = nn
        self.vb = vb
        self.lang = lang

    def do_augment(self, doc: model.Document):
        pos_tags = []
        #  put the wanted POS Tags in the list, to filter
        if self.adj_adv:
            pos_tags.extend(["JJ", "JJS", "JJR", "RB", "RBR", "RBS"])
        if self.nn:
            pos_tags.extend(["NN", "NNS", "NNP", "NNPS"])
        if self.vb:
            pos_tags.extend(["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"])
        for sentence in doc.sentences:
            i = 0
            while i < len(sentence.tokens):
                token = sentence.tokens[i]
                if token.pos_tag in pos_tags:
                    translated_text = self.back_translate(token.text)
                    translated_list = translated_text.split()
                    token.text = translated_list[0].lower()
                    token.pos_tag = tokenmanager.get_pos_tag([translated_list[0].lower()])[0]
                    #  when the returned text has more than 1 word, new tokens must be created
                    if len(translated_list) > 1:
                        print(translated_list)
                        m = copy.deepcopy(i)
                        for j in range(1, len(translated_list)):
                            if m + j < len(sentence.tokens):
                                tok = model.Token(text=translated_list[j],
                                                  index_in_document=sentence.tokens[m + j].index_in_document,
                                                  pos_tag=tokenmanager.get_pos_tag([translated_list[j].lower()])[0],
                                                  bio_tag=tokenmanager.get_bio_tag_based_on_left_token(token.bio_tag),
                                                  sentence_index=token.sentence_index)
                                tokenmanager.create_token(doc, tok, m + j)
                                i += 1
                i += 1
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
