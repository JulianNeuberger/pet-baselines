from augment import base
from data import model
from transformations import tokenmanager
import typing
from random import random as rand
from random import choice
# Filler Word Augmentation - Rauschen


class Trafo40Step(base.AugmentationStep):

    def __init__(self, prob=0.166, speaker_p:bool=True, uncertain_p:bool =True, filler_p:bool =True, tags: typing.List = None):
        self.prob = prob
        self.speaker_p = speaker_p
        self.uncertain_p = uncertain_p
        self.filler_p = filler_p
        self.tags = tags

    def do_augment(self, doc: model.Document) -> model.Document:
        doc = doc.copy()
        for mention in doc.mentions:
            assert max(mention.token_indices) < len(doc.sentences[mention.sentence_index].tokens), "broken before trafo 40"
        # Speaker opinion/mental state phrases
        # Taken from Kovatchev et al. (2021)
        speaker_phrases = [
            "I think",
            "I believe",
            "I mean",
            "I guess",
            "that is",
            "I assume",
            "I feel",
            "In my opinion",
            "I would say",
        ]

        # Words and phrases indicating uncertainty
        # Taken from Kovatchev et al. (2021)
        uncertain_phrases = [
            "maybe",
            "perhaps",
            "probably",
            "possibly",
            "most likely",
        ]

        # Filler words that should preserve the meaning of the phrase
        # Taken from Laserna et al. (2014)
        fill_phrases = [
            "uhm",
            "umm",
            "ahh",
            "err",
            "actually",
            "obviously",
            "naturally",
            "like",
            "you know",
        ]

        # Initialize the list of all augmentation phrases
        all_fill = []

        # Add speaker phrases, if enabled
        if self.speaker_p:
            all_fill += speaker_phrases

        # Add uncertain phrases, if enabled
        if self.uncertain_p:
            all_fill += uncertain_phrases

        # Add filler phrases, if enabled
        if self.filler_p:
            all_fill += fill_phrases
        for sentence_counter, sentence in enumerate(doc.sentences):
            # token counter to determine the sentence index and skip inserted words
            token_counter = 0

            while token_counter < len(sentence.tokens) - 1:  # -1 so that nothing can be inserted after the point
                token = sentence.tokens[token_counter]
                # only if tags are given the tokens should be checked for tags
                if self.tags is not None:
                    if token.bio_tag not in self.tags:
                        token_counter += 1
                        continue

                if rand() <= self.prob:
                    # choose filler phrase
                    random_filler = choice(all_fill).split()

                    # generate bio-tag
                    bio_tag = tokenmanager.get_bio_tag_based_on_left_token(token.bio_tag)

                    # get mention_index
                    mentions = tokenmanager.get_mentions(doc, token_counter, sentence_counter)

                    # set index_in_sentence
                    index_in_sentence = token_counter
                    for i in range(0, len(random_filler)):
                        # create tokens per inserted word
                        tok = model.Token(text=random_filler[i], index_in_document=token.index_in_document + i + 1,
                                          pos_tag=tokenmanager.get_pos_tag([random_filler[i]])[0], bio_tag=bio_tag,
                                          sentence_index=token.sentence_index)
                        if mentions == []:
                            tokenmanager.create_token(doc, tok, index_in_sentence + i + 1, None)
                        else:
                            tokenmanager.create_token(doc, tok, index_in_sentence + i + 1, mentions[0])

                        # increase the token_counter so that the inserted words get skipped
                        token_counter += 1

                token_counter += 1
        for mention in doc.mentions:
            assert max(mention.token_indices) < len(doc.sentences[mention.sentence_index].tokens), "broken after trafo 40"
        return doc