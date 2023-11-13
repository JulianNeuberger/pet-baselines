import copy
import typing
from random import random

from transformers import FSMTForConditionalGeneration, FSMTTokenizer

from augment import base, params
from data import model
from transformations import tokenmanager


# Author: Benedikt
class Trafo8Step(base.AugmentationStep):
    def __init__(self, p: float = 1, lang: str = "de", **kwargs):
        super().__init__(**kwargs)
        self.lang = lang
        self.p = p
        self.max_outputs = 1
        self.num_beams = 2
        name_en_de = "facebook/wmt19-en-de"
        self.tokenizer_en_de = FSMTTokenizer.from_pretrained(name_en_de)
        self.model_en_de = FSMTForConditionalGeneration.from_pretrained(name_en_de)
        name_de_en = "facebook/wmt19-de-en"
        self.tokenizer_de_en = FSMTTokenizer.from_pretrained(name_de_en)
        self.model_de_en = FSMTForConditionalGeneration.from_pretrained(name_de_en)

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.FloatParam(name="p", min_value=0, max_value=1),
            params.IntegerParam(name="max_outputs", min_value=1, max_value=10),
            params.IntegerParam(name="num_beams", min_value=1, max_value=20),
        ]

    def do_augment(self, doc2: model.Document):
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
            de = self.en2de(en)
            en_new = self.de2en(de)
        except Exception as ex:
            print(ex)
            print("Returning Default due to Run Time Exception")
            en_new = en
        return en_new

    def en2de(self, input):
        input_ids = self.tokenizer_en_de.encode(input, return_tensors="pt")
        outputs = self.model_en_de.generate(input_ids)
        decoded = self.tokenizer_en_de.decode(outputs[0], skip_special_tokens=True)
        return decoded

    def de2en(self, input):
        input_ids = self.tokenizer_de_en.encode(input, return_tensors="pt")
        outputs = self.model_de_en.generate(
            input_ids,
            num_return_sequences=self.max_outputs,
            num_beams=self.num_beams,
        )
        predicted_outputs = []
        for output in outputs:
            decoded = self.tokenizer_de_en.decode(output, skip_special_tokens=True)
            # TODO: this should be able to return multiple sequences
            predicted_outputs.append(decoded)
        return predicted_outputs[0]


# step = Trafo8Step()
# print(step.back_translate("managing"))
