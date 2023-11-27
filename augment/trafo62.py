import typing

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from augment import base, params
from data import model
from transformations import tokenmanager

from nltk import tokenize


class Trafo62Step(base.AugmentationStep):
    languages = [
        "af",
        "am",
        "ar",
        "ast",
        "az",
        "ba",
        "be",
        "bg",
        "bn",
        "br",
        "bs",
        "ca",
        "ceb",
        "cs",
        "cy",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "fa",
        "ff",
        "fi",
        "fr",
        "fy",
        "ga",
        "gd",
        "gl",
        "gu",
        "ha",
        "he",
        "hi",
        "hr",
        "ht",
        "hu",
        "hy",
        "id",
        "ig",
        "ilo",
        "is",
        "it",
        "ja",
        "jv",
        "ka",
        "kk",
        "km",
        "kn",
        "ko",
        "lb",
        "lg",
        "ln",
        "lo",
        "lt",
        "lv",
        "mg",
        "mk",
        "ml",
        "mn",
        "mr",
        "ms",
        "my",
        "ne",
        "nl",
        "no",
        "ns",
        "oc",
        "or",
        "pa",
        "pl",
        "ps",
        "pt",
        "ro",
        "ru",
        "sd",
        "si",
        "sk",
        "sl",
        "so",
        "sq",
        "sr",
        "ss",
        "su",
        "sv",
        "sw",
        "ta",
        "th",
        "tl",
        "tn",
        "tr",
        "uk",
        "ur",
        "uz",
        "vi",
        "wo",
        "xh",
        "yi",
        "yo",
        "zh",
        "zu",
    ]

    def __init__(self, dataset: typing.List[model.Document], pivot_language: str = "de"):
        super().__init__(dataset)
        self.model = M2M100ForConditionalGeneration.from_pretrained(
            "facebook/m2m100_418M"
        )
        self.tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        self.src_lang = "en"
        self.pivot_lang = pivot_language

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.ChoiceParam(name="pivot_language", choices=Trafo62Step.languages),
        ]

    def do_augment(self, doc: model.Document):
        doc = doc.copy()
        candidates = self.get_sequences(doc)

        for candidate in candidates:
            sentence = " ".join(t.text for t in candidate)
            translated_candidate = self.back_translate(sentence)[0]
            translated_tokens = tokenize.word_tokenize(translated_candidate)
            print("Input : ", sentence)
            print("Output: ", " ".join(translated_tokens))
            sentence_id = candidate[0].sentence_index
            start_in_sentence = candidate[0].index_in_sentence(doc)
            stop_in_sentence = candidate[-1].index_in_sentence(doc) + 1
            tokenmanager.replace_sequence_text_in_sentence(
                doc, sentence_id, start_in_sentence, stop_in_sentence, translated_tokens
            )
        return doc

    def back_translate(self, sentence: str):
        pivot_sentence = self.translate(sentence, self.src_lang, self.pivot_lang, self.model, self.tokenizer)
        return self.translate(pivot_sentence, self.pivot_lang, self.src_lang, self.model, self.tokenizer)

    @staticmethod
    def translate(sentence: str, source_lang: str, target_lang: str, translation_model, tokenizer) -> str:
        tokenizer.src_lang = source_lang
        encoded_source_sentence = tokenizer(sentence, return_tensors="pt")
        generated_target_tokens = translation_model.generate(
            **encoded_source_sentence,
            forced_bos_token_id=tokenizer.get_lang_id(target_lang)
        )
        target_sentence = tokenizer.batch_decode(
            generated_target_tokens, skip_special_tokens=True
        )
        return target_sentence
