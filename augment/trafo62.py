import typing
from random import random

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from augment import base, params
from data import model
from transformations import tokenmanager


class Trafo62Step(base.AugmentationStep):
    def __init__(
        self, dataset: typing.List[model.Document], p: float = 1, lang: str = "de"
    ):
        super().__init__(dataset)
        self.model = M2M100ForConditionalGeneration.from_pretrained(
            "facebook/m2m100_418M"
        )
        self.tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        self.src_lang = "en"
        self.pivot_lang = lang
        self.p = p

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.FloatParam(name="p", min_value=0, max_value=1),
            params.ChoiceParam(name="lang", choices=["de", "fr", "es"]),
        ]

    def do_augment(self, doc: model.Document):
        doc = doc.copy()
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
                    translated = self.back_translate(en=text_before)
                text_before_list = text_before.split()
                translated_list = translated[0].split()
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
                                bio_tag=tokenmanager.get_bio_tag_based_on_left_token(
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

    #  returns the back translated text, when it's not working, it returns the old text
    def back_translate(self, en: str):
        try:
            de = self.translate(en, self.pivot_lang, self.model, self.tokenizer)
            en_new = self.translate(de, self.src_lang, self.model, self.tokenizer)
        except Exception as ex:
            print(ex)
            print("Returning Default due to Run Time Exception")
            en_new = en
        return en_new

    def translate(self, sentence, target_lang, model, tokenizer):
        tokenizer.src_lang = self.src_lang
        encoded_source_sentence = tokenizer(sentence, return_tensors="pt")
        generated_target_tokens = model.generate(
            **encoded_source_sentence,
            forced_bos_token_id=tokenizer.get_lang_id(target_lang)
        )
        target_sentence = tokenizer.batch_decode(
            generated_target_tokens, skip_special_tokens=True
        )
        return target_sentence
