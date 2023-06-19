import copy
from random import random

from transformers import FSMTForConditionalGeneration, FSMTTokenizer
from transformers import pipeline
from data import model
from transformations import tokenmanager


class LostInTranslation:
    def __init__(self, adj_adv: bool, nn: bool, vb: bool, p: float = 1, lang: str = "de"):
        self.adj_adv = adj_adv
        self.nn = nn
        self.vb = vb
        self.lang = lang
        self.p = p

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

    def generate(self, doc: model.Document):
        for sentence in doc.sentences:
            i = 0
            while i < len(sentence.tokens) - 1:
                token = sentence.tokens[i]
                current_bio = tokenmanager.get_bio_tag_short(token.bio_tag)
                text_before = ""
                j = 0
                print("-----------")
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
                translated = self.back_translate(text_before)
                text_before_list = text_before.split()
                translated_list = translated.split()
                diff = len(translated_list) - len(text_before_list)
                print(diff)
                print(text_before_list)
                print(translated_list)

                if diff > 0:
                    token.text = translated_list[0]
                    token.pos_tag = tokenmanager.get_pos_tag([token.text])[0]
                    if len(translated_list) > 1:
                        for k in range(1, len(translated_list)):
                            tok = model.Token(text=translated_list[k], index_in_document=token.index_in_document + i + k,
                                        pos_tag=tokenmanager.get_pos_tag([token.text])[0], bio_tag=tokenmanager.get_bio_tag_based_on_left_token(token.bio_tag),
                                        sentence_index=token.sentence_index)
                            tokenmanager.create_token(doc, tok, i + k)
                elif diff == 0:
                    for k in range(0, len(translated_list)):
                        print(i)
                        index_in_doc = token.index_in_document
                        sentence.tokens[i + k].text = translated_list[k]
                        sentence.tokens[i + k].pos_tag = tokenmanager.get_pos_tag([token.text])[0]
                else:
                    for k in range(len(translated_list)):
                        sentence.tokens[i + k].text = translated_list[k]
                        sentence.tokens[i + k].pos_tag = tokenmanager.get_pos_tag([token.text])[0]
                    for k in range(len(translated_list), len(text_before_list)):
                        tokenmanager.delete_token(doc, sentence.tokens[i + len(translated_list)].index_in_document)
                #print(translated)
                #translated_text = self.back_translate(token.text)
                i = i + j + diff
        return doc


tokens = [model.Token(text="University", index_in_document=0,
                      pos_tag="PRP", bio_tag="B-Actor",
                      sentence_index=0), model.Token(text="member", index_in_document=1,
                                                     pos_tag="VBP", bio_tag="O",
                                                     sentence_index=0), model.Token(text="legal", index_in_document=2,
                                                                                    pos_tag="NN",
                                                                                    bio_tag="B-Activity Data",
                                                                                    sentence_index=0),
          model.Token(text="advice", index_in_document=3,
                      pos_tag="NNS", bio_tag="I-Activity Data",
                      sentence_index=0),
model.Token(text="advice", index_in_document=4,
                      pos_tag="NNS", bio_tag="I-Activity Data",
                      sentence_index=0),
model.Token(text="advice", index_in_document=5,
                      pos_tag="NNS", bio_tag="I-Activity Data",
                      sentence_index=0),
          model.Token(text="easy", index_in_document=6,
                      pos_tag="JJ", bio_tag="O",
                      sentence_index=0),
model.Token(text="task", index_in_document=7,
                      pos_tag="JJ", bio_tag="O",
                      sentence_index=0),
          model.Token(text=".", index_in_document=8,
                      pos_tag=".", bio_tag=".",
                      sentence_index=0)
          ]

sentence1 = model.Sentence(tokens=tokens)

doc = model.Document(
    text="I leave HR office",
    name="1", sentences=[sentence1],
    mentions=[],
    entities=[],
    relations=[])

trafo = LostInTranslation(True, True, True, 1, "de")
print(trafo.generate(doc))
