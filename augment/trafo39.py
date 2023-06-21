from augment import base
from data import model
from transformations import tokenmanager
from random import choice
from random import random as rand
from numpy.random import binomial

# English Mention Replacement for Ner


class Trafo39Step(base.AugmentationStep):

    def __init__(self, prob=0.5):
        self.prob = prob

    def do_augment(self, doc: model.Document):

        # prepare token and tag sequences
        token_sequences = []
        tag_sequences = []
        entity_mentions_by_type = {}
        for sentence in doc.sentences:
            token_text = []
            token_tags = []
            for token in sentence.tokens:
                token_text.append(token.text)
                token_tags.append(token.bio_tag)
            token_sequences.append(token_text)
            tag_sequences.append(token_tags)

        entity_mentions_by_type = Trafo39Step.extract_entity_mentions_by_type(self, token_sequences, tag_sequences)

        # change mentions in sentences
        sentence_idx = 0
        changes_counter = 0
        for sentence in doc.sentences:
            changes_counter = 0
            assert len(token_sequences[sentence_idx]) == len(tag_sequences[
                                                                 sentence_idx]), f"token_sequence and tag_sequence should have same length! {len(token_sequences[sentence_idx])}!={len(tag_sequences[sentence_idx])}"

            # get the entities of the sentence
            entities = Trafo39Step.extract_entities(self, token_sequences[sentence_idx], tag_sequences[sentence_idx])

            # only if the sentence has entities they can be changed
            if len(entities) == 0:
                pass
            else:
                # for each entity determine whether it should be changed and if yes change it
                for entity in entities:
                    assert (entity[
                                "type"] in entity_mentions_by_type), f"invalid entity type {entity['type']} in tag_sequence"
                    if rand() <= self.prob:

                        # choose the new entity text out of all the texts belonging to this entity
                        new_text = choice(entity_mentions_by_type[entity["type"]])
                        token_texts = new_text.split()

                        # get the index in document from where to start changing the tokens
                        start_index_in_doc = sentence.tokens[entity["start"] + changes_counter].index_in_document

                        # new text
                        tokens = []
                        # create a new token per word
                        for i in range(len(token_texts)):
                            tokens.append(model.Token(text=token_texts[i], index_in_document=start_index_in_doc + i,
                                                      pos_tag=tokenmanager.get_pos_tag([token_texts[i]])[0],
                                                      bio_tag="",
                                                      sentence_index=sentence_idx))

                            # set bio-tags
                            if i == 0:
                                tokens[i].bio_tag = "B-" + entity["type"]
                            else:
                                tokens[i].bio_tag = "I-" + entity["type"]

                        # set the mention - if new_mention is True: id from new text, False: from old
                        # original text
                        entity_text = entity["text"].split(" ")

                        # set index_in_sentence from tokens that get inserted
                        ind_in_sent = tokenmanager.get_index_in_sentence(sentence, entity_text, start_index_in_doc)

                        # search for mentions where the token is listed
                        mentions = tokenmanager.get_mentions(doc, ind_in_sent, sentence_idx)
                        for i in range(len(tokens)):
                            if i < len(entity_text):
                                sentence.tokens[ind_in_sent + i] = tokens[i]
                            if i >= len(entity_text):
                                tokenmanager.create_token(doc, tokens[i], ind_in_sent + i, mentions[0])
                                changes_counter += 1
                        if len(entity_text) > len(tokens):
                            num_of_changes = len(entity_text) - len(tokens)
                            for i in range(num_of_changes):
                                tokenmanager.delete_token(doc, start_index_in_doc + len(tokens))
                                changes_counter -= 1
            sentence_idx += 1
        return doc

    def extract_entity_mentions_by_type(self, token_sequences, tag_sequences):
        """
        Extract entity mentions categorized by entity type from a list of token_sequences and tags tag_sequences.
        Es werden die Mentions ausfindig gemacht und als "Entität" mit Text, Tag, Start und Ende gespeichert.
        """
        entity_mention_by_type = {}
        for sent_tokens, sent_tags in zip(token_sequences, tag_sequences):
            sent_entities = Trafo39Step.extract_entities(self,
                sent_tokens, sent_tags
            )
            for entity in sent_entities:
                if entity["type"] not in entity_mention_by_type:
                    entity_mention_by_type[entity["type"]] = set()
                entity_mention_by_type[entity["type"]].add(entity["text"])
        for entity_type in entity_mention_by_type:
            entity_mention_by_type[entity_type] = list(
                entity_mention_by_type[entity_type]
            )
        return entity_mention_by_type

    def extract_entities(self, tokens, tags):
        entities = []
        last_tag = "O"
        phrase_tokens = []
        start_idx, end_idx = -1, -1
        for idx in range(len(tokens)):
            token, tag = tokens[idx], tags[idx]
            start_chunk, end_chunk = False, False
            appended = False
            end_token = idx == len(tags) - 1
            if (last_tag == "O" and tag.startswith("B")) or (
                    last_tag == "O" and tag.startswith("I")
            ):
                start_chunk = True
                end_chunk = False
                start_idx = idx
            if (last_tag.startswith("B") and tag == "O") or (
                    last_tag.startswith("I") and tag == "O"
            ):
                start_chunk = False
                end_chunk = True
                end_idx = idx
            if (
                    (last_tag.startswith("B") and tag.startswith("I"))
                    or (last_tag.startswith("I") and tag.startswith("I"))
                    or (last_tag == "O" and tag == "O")
            ):
                start_chunk = False
                end_chunk = False
            if (last_tag.startswith("I") and tag.startswith("B")) or (
                    last_tag.startswith("B") and tag.startswith("B")
            ):
                start_chunk = True
                end_chunk = True
                end_idx = idx

            if start_chunk and end_chunk:
                phrase_str = " ".join(phrase_tokens)
                entity = {
                    "text": phrase_str,
                    "type": last_tag.split("-")[-1],
                    "start": start_idx,
                    "end": end_idx,
                }
                entities.append(entity)
                phrase_tokens = []
                start_chunk = False
                end_chunk = False
                start_idx = idx

            if start_chunk:
                phrase_tokens.append(token)
                start_chunk = False
                appended = True

            if end_token and tag != "O":
                if not appended:
                    phrase_tokens.append(token)
                    appended = True
                if not end_chunk:
                    last_tag = tag
                    end_chunk = True
                    end_idx = idx

            if end_chunk:
                phrase_str = " ".join(phrase_tokens)
                entity = {
                    "text": phrase_str,
                    "type": last_tag.split("-")[-1],
                    "start": start_idx,
                    "end": end_idx,
                }
                entities.append(entity)
                phrase_tokens = []
                end_chunk = False

            # middle of the entity mention
            if last_tag != "O" and tag != "O":
                if not appended:
                    phrase_tokens.append(token)
            last_tag = tag

        return entities


    #def format_entity(self, entity_text, entity_type):
        #"""
        #Format the entity type tags in the correct BIO format.
        #"""
        #entity_tokens = entity_text.split()
        #entity_tags = [f"I-{entity_type}"] * len(entity_tokens)
        #entity_tags[0] = f"B-{entity_type}"

       # return entity_tokens, entity_tags
