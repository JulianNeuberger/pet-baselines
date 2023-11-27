import copy
import typing
from random import random

from transformers import pipeline

from data import model
from augment import base, params
from transformations import tokenmanager


# Author: Benedikt
class Trafo58Step(base.AugmentationStep):
    def __init__(
        self,
        dataset: typing.List[model.Document],
        p: float = 1,
        lang: str = "de",
    ):
        super().__init__(dataset)
        self.lang = lang
        self.p = p
        self.encode = pipeline(
            "translation_en_to_{}".format(self.lang),
            model="Helsinki-NLP/opus-mt-en-{}".format(self.lang),
            device="cuda:0",
        )
        self.decode = pipeline(
            "translation_{}_to_en".format(self.lang),
            model="Helsinki-NLP/opus-mt-{}-en".format(self.lang),
            device="cuda:0",
        )
        self.vocab = self.decode.tokenizer.get_vocab()

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.FloatParam(name="p", min_value=0.0, max_value=1.0),
            params.ChoiceParam(name="lang", choices=["de", "fr", "es"]),
        ]

    def do_augment(self, doc: model.Document) -> model.Document:
        return self.do_augment_batch_wise(doc)

    def do_augment_batch_wise(self, doc: model.Document):
        doc = copy.deepcopy(doc)
        to_translate = []
        for mention in doc.mentions:
            if random() >= self.p:
                continue
            to_translate.append(mention)
        to_translate_text = [e.text(doc) for e in to_translate]
        translated_mentions = self.back_translate_batch(to_translate_text)
        for mention, new_mention_text in zip(to_translate, translated_mentions):
            new_mention_tokens = new_mention_text.split(" ")
            tokenmanager.replace_mention_text(
                doc, doc.mention_index(mention), new_mention_tokens
            )
        return doc

    def do_augment_single(self, doc2: model.Document):
        doc = copy.deepcopy(doc2)
        for sentence in doc.sentences:
            i = 0
            while i < len(sentence.tokens) - 1:
                token = sentence.tokens[i]
                current_bio = tokenmanager.get_bio_tag_short(token.bio_tag)
                text_before = ""
                j = 0
                while (
                    tokenmanager.get_bio_tag_short(sentence.tokens[i + j].bio_tag)
                    == current_bio
                ):
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
                    print(f"Translated {text_before} to {translated}")
                text_before_list = text_before.split()
                translated_list = translated.split()
                diff = len(translated_list) - len(text_before_list)
                if diff > 0:
                    token.text = translated_list[0]
                    token.pos_tag = tokenmanager.get_pos_tag([token.text])[0]
                    if len(translated_list) > 1:
                        for k in range(1, len(translated_list)):
                            tok = model.Token(
                                text=translated_list[k],
                                index_in_document=token.index_in_document + i + k,
                                pos_tag=tokenmanager.get_pos_tag([token.text])[0],
                                bio_tag=tokenmanager.get_continued_bio_tag(
                                    token.bio_tag
                                ),
                                sentence_index=token.sentence_index,
                            )
                            tokenmanager.create_token(doc, tok, i + k)
                elif diff == 0:
                    for k in range(0, len(translated_list)):
                        index_in_doc = token.index_in_document
                        sentence.tokens[i + k].text = translated_list[k]
                        sentence.tokens[i + k].pos_tag = tokenmanager.get_pos_tag(
                            [token.text]
                        )[0]
                else:
                    for k in range(len(translated_list)):
                        sentence.tokens[i + k].text = translated_list[k]
                        sentence.tokens[i + k].pos_tag = tokenmanager.get_pos_tag(
                            [token.text]
                        )[0]
                    for k in range(len(translated_list), len(text_before_list)):
                        tokenmanager.delete_token(
                            doc,
                            sentence.tokens[i + len(translated_list)].index_in_document,
                        )
                i = i + j + diff
        return doc

    def back_translate_batch(self, en_batch: typing.List[str]) -> typing.List[str]:
        try:
            return self.encode_decode(en_batch)
        except RuntimeError:
            print(f"Returning Default due to Run Time Exception in batch {en_batch}")
            return en_batch

    #  returns the back translated text, when it's not working, it returns the old text
    def back_translate(self, en: str):
        try:
            en_new = self.encode_decode([en])[0]
        except Exception as ex:
            print(ex)
            print("Returning Default due to Run Time Exception")
            en_new = en
        return en_new

    def encode_decode(self, texts: typing.List[str]) -> typing.List[str]:
        # translate and un-translate
        # using Helsinki-NLP OpusMT models
        # en->lang->en
        encoded = [t["translation_text"] for t in self.encode(texts, max_length=600)]
        decoded = [t["translation_text"] for t in self.decode(encoded, max_length=600)]
        print(f"Translated {texts} to {decoded}.")
        return decoded
        # return self.decode(
        #     self.encode(text, max_length=600)[0]["translation_text"],
        #     max_length=600,
        # )[0]["translation_text"]
