import typing
from random import random as rand

from nltk.corpus import wordnet

from augment import base, params
from data import model
from transformations import tokenmanager


# Adjective Antonym Switch - Wortebene


# Author: Leonie
class Trafo3Step(base.AugmentationStep):
    def __init__(
        self,
        dataset: typing.List[model.Document],
        no_dupl: bool = False,
        max_adj: int = 1,
        prob: float = 0.5,
        **kwargs
    ):
        super().__init__(dataset)
        self.max_adj = max_adj
        self.no_dupl = no_dupl
        self.prob = prob

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [
            params.BooleanParameter(name="no_dupl"),
            params.IntegerParam(name="max_adj", min_value=0, max_value=20),
            params.FloatParam(name="prob", min_value=0.0, max_value=1.0),
        ]

    def do_augment(self, doc: model.Document) -> model.Document:
        doc = doc.copy()
        changed_adjectives = []  # contains all changed adjectives

        for sentence in doc.sentences:
            num_of_changes = 0
            j = 0
            while j < len(sentence.tokens):
                token = sentence.tokens[j]

                # if the adjective already has been changed, exists is set to True
                exists = False
                if (
                    token.pos_tag in ["JJ", "JJS", "JJR"]
                    and (num_of_changes < self.max_adj)
                    and rand() <= self.prob
                ):
                    # in case of no_dupl == True set exists = True if token.text is in changed_adjectives
                    if self.no_dupl is True and (token.text in changed_adjectives):
                        exists = True

                    # in case of no_dupl == True, a token will only be changed if it hasn't been changed before
                    # in case of no_dupl == False, a token will always be changed
                    if not exists:
                        # Get Synsets
                        synsets = wordnet.synsets(token.text, "a")
                        antonyms = []

                        # Get Antonyms
                        if synsets:
                            first_synset = synsets[0]
                            lemmas = first_synset.lemmas()
                            first_lemma = lemmas[0]
                            antonyms = first_lemma.antonyms()

                        # Get first Antonym
                        if antonyms:
                            changed_adjectives.append(token.text)
                            antonyms.sort(key=lambda x: str(x).split(".")[2])
                            first_antonym = antonyms[0].name()
                            first_antonym = first_antonym.split("_")

                            # Replace adjective with antonym
                            token.text = first_antonym[0]
                            if len(first_antonym) > 1:
                                ment_ind = tokenmanager.get_mentions(
                                    doc, j, token.sentence_index
                                )
                                for i in range(1, len(first_antonym)):
                                    tok = model.Token(
                                        text=first_antonym[i],
                                        index_in_document=token.index_in_document + i,
                                        pos_tag=tokenmanager.get_pos_tag(
                                            [first_antonym[i]]
                                        )[0],
                                        bio_tag=tokenmanager.get_bio_tag_based_on_left_token(
                                            token.bio_tag
                                        ),
                                        sentence_index=token.sentence_index,
                                    )
                                    if ment_ind == []:
                                        tokenmanager.create_token(doc, tok, j + i, None)
                                    else:
                                        tokenmanager.create_token(
                                            doc, tok, j + i, ment_ind[0]
                                        )
                                    j += 1
                            num_of_changes += 1
                j += 1
        return doc
