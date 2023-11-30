import typing

from nltk import tokenize
from transformers import FSMTForConditionalGeneration, FSMTTokenizer

from augment import base, params
from data import model
from transformations import tokenmanager


# Author: Benedikt
class Trafo8Step(base.BaseTokenReplacementStep):
    def __init__(
        self,
        dataset: typing.List[model.Document],
        n: int = 1,
        num_beams: int = 2,
        lang: str = "de",
    ):
        super().__init__(dataset, n)
        self.lang = lang
        self.num_beams = num_beams
        name_en_de = "facebook/wmt19-en-de"
        self.tokenizer_en_de = FSMTTokenizer.from_pretrained(name_en_de)
        self.model_en_de = FSMTForConditionalGeneration.from_pretrained(name_en_de).to(
            "cuda"
        )
        name_de_en = "facebook/wmt19-de-en"
        self.tokenizer_de_en = FSMTTokenizer.from_pretrained(name_de_en)
        self.model_de_en = FSMTForConditionalGeneration.from_pretrained(name_de_en).to(
            "cuda"
        )

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.IntegerParam(name="n", min_value=1, max_value=20),
            params.IntegerParam(name="num_beams", min_value=1, max_value=20),
        ]

    def do_augment(self, doc: model.Document) -> model.Document:
        doc = doc.copy()

        candidates = self.get_replacement_candidates(doc)
        num_changes = 0
        for candidate in candidates:
            text = " ".join(t.text for t in candidate)
            if text in [",", ".", "?", "!", ":", "#", "-"]:
                continue

            translated = self.back_translate(text)
            if translated is None:
                continue

            translated_tokens = tokenize.word_tokenize(translated)
            tokenmanager.replace_sequence_text_in_sentence(
                doc,
                candidate[0].sentence_index,
                candidate[0].index_in_sentence(doc),
                candidate[-1].index_in_sentence(doc),
                translated_tokens,
            )

            num_changes += 1
            if num_changes == self.n:
                break

        return doc

    def back_translate(self, en: str) -> typing.Optional[str]:
        try:
            de = self.en2de(en)
            return self.de2en(de)
        except Exception as ex:
            print(ex)
            print("Returning Default due to Run Time Exception")
            return None

    def en2de(self, input):
        input_ids = self.tokenizer_en_de.encode(input, return_tensors="pt").to("cuda")
        outputs = self.model_en_de.generate(input_ids)
        decoded = self.tokenizer_en_de.decode(
            outputs[0].to("cuda"), skip_special_tokens=True
        )
        return decoded

    def de2en(self, input):
        input_ids = self.tokenizer_de_en.encode(input, return_tensors="pt").to("cuda")
        outputs = self.model_de_en.generate(
            input_ids,
            num_return_sequences=self.max_outputs,
            num_beams=self.num_beams,
        )
        predicted_outputs = []
        for output in outputs:
            decoded = self.tokenizer_de_en.decode(
                output.to("cuda"), skip_special_tokens=True
            )
            # TODO: this should be able to return multiple sequences
            predicted_outputs.append(decoded)
        return predicted_outputs[0]
