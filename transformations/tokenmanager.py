from data import model, Document
import typing
import nltk
import data


# Delete a Token #


# Author: Benedikt
def delete_token(doc: Document, index_in_document: int):
    # delete token from sentence.tokens, get position from where to change the following tokens
    index = delete_token_from_tokens(doc, index_in_document)
    index_in_sentence = index[0]
    sentence_index = index[1]
    mention_index = delete_token_from_mention_token_indices(
        doc, index_in_sentence, sentence_index
    )


# Author: Benedikt
def delete_token_from_tokens(
    doc: Document, index_in_document: int
) -> typing.List:  # deletes tokens from the tokenslist
    not_found_token = True
    index = []
    for sentence in doc.sentences:
        counter = 0
        for token in sentence.tokens:
            if token.index_in_document == index_in_document and not_found_token:
                not_found_token = False
                # delete token from sentence.tokens
                del sentence.tokens[counter]
                # get the position from where to change the index_in_document for all tokens and the sentence_index
                if counter < len(sentence.tokens):
                    sentence.tokens[counter].index_in_document -= 1
                index.append(counter)
                index.append(token.sentence_index)
            # change index_in_doc of following tokens
            if not not_found_token:
                token.index_in_document -= 1
            counter += 1
    return index


# Author: Benedikt
def delete_token_from_mention_token_indices(
    doc: Document, token_index_in_sentence: int, sentence_index: int
):
    mention_to_delete = None
    for mention_id, mention in enumerate(doc.mentions):
        if mention.sentence_index != sentence_index:
            continue

        if token_index_in_sentence in mention.token_indices:
            if len(mention.token_indices) == 1:
                doc.mentions.remove(mention)
                delete_mention_from_entities(doc, mention_id)
                adjust_mention_indices_in_entities(doc, mention_id)
                mention_to_delete = mention_id
            else:
                mention.token_indices.remove(token_index_in_sentence)
                for i in range(len(mention.token_indices)):
                    if mention.token_indices[i] > token_index_in_sentence:
                        mention.token_indices[i] -= 1
        else:
            for i in range(len(mention.token_indices)):
                if mention.token_indices[i] > token_index_in_sentence:
                    mention.token_indices[i] -= 1
        mention_id += 1
    return mention_to_delete


def adjust_mention_indices_in_entities(doc: Document, mention_index: int):
    for entity in doc.entities:
        for i in range(len(entity.mention_indices)):
            if entity.mention_indices[i] > mention_index:
                entity.mention_indices[i] -= 1


def adjust_entity_indices_in_relations(doc: Document, entity_index: int):
    for relation in doc.relations:
        if relation.head_entity_index > entity_index:
            relation.head_entity_index -= 1

        if relation.tail_entity_index > entity_index:
            relation.tail_entity_index -= 1


def delete_mention_from_entities(doc: Document, mention_index: int):
    for entity_id, entity in enumerate(doc.entities):
        if mention_index not in entity.mention_indices:
            continue

        is_last_mention = len(entity.mention_indices) == 1

        if is_last_mention:
            doc.entities.remove(entity)
            delete_relations(doc, entity_id)
            adjust_entity_indices_in_relations(doc, entity_id)
            break
        else:
            entity.mention_indices.remove(mention_index)


def delete_relations(doc: Document, entity_index: int):
    affected_relation_indices = set()
    for relation_index, relation in enumerate(doc.relations):
        if relation.head_entity_index == entity_index:
            affected_relation_indices.add(relation_index)
        if relation.tail_entity_index == entity_index:
            affected_relation_indices.add(relation_index)
    for relation_index in sorted(list(affected_relation_indices), reverse=True):
        doc.relations.pop(relation_index)


# CREATE A NEW TOKEN #
# Author: Leonie
def create_token(
    doc: Document,
    token: model.Token,
    index_in_sentence: int,
    mention_index=None,
):  # insert a token
    # only if index in sentence and mention id fit should changes be made
    if 0 <= index_in_sentence < len(doc.sentences[token.sentence_index].tokens):
        if mention_index is not None:
            if 0 <= mention_index < len(doc.mentions):
                # a new given Token gets inserted in the token array of the belonging sentence and all index_in_doc are adjusted
                insert_token_in_tokens(doc, token, index_in_sentence)
                # adds token_id to mention if a mention id is given
                insert_token_in_mentions(doc, index_in_sentence, mention_index)
        else:
            # a new given Token gets inserted in the token array of the belonging sentence and all index_in_doc are adjusted
            insert_token_in_tokens(doc, token, index_in_sentence)


def replace_mention_text(
    doc: Document, mention_index: int, new_token_texts: typing.List[str]
):
    mention = doc.mentions[mention_index]
    new_pos_tags = get_pos_tag(new_token_texts)
    new_bio_tags = [f"B-{mention.ner_tag}"] + (len(new_token_texts) - 1) * [
        f"I-{mention.ner_tag}"
    ]
    mention_start = mention.document_level_token_indices(doc)[0]

    new_tokens = [
        model.Token(
            text=new_text,
            pos_tag=new_pos,
            index_in_document=mention_start + new_index,
            bio_tag=new_bio_tag,
            sentence_index=mention.sentence_index,
        )
        for new_text, new_pos, new_index, new_bio_tag in zip(
            new_token_texts, new_pos_tags, range(len(new_token_texts)), new_bio_tags
        )
    ]
    old_tokens = [
        doc.sentences[mention.sentence_index].tokens[i] for i in mention.token_indices
    ]

    length_difference = len(new_tokens) - len(old_tokens)

    sentence_level_mention_start = mention.token_indices[0]

    if length_difference == 0:
        for i, token in enumerate(new_tokens):
            doc.sentences[mention.sentence_index].tokens[
                sentence_level_mention_start + i
            ] = token
        return

    if length_difference < 0:
        for old_token in old_tokens[len(new_tokens) :]:
            delete_token(doc, old_token.index_in_document)
        for i, token in enumerate(new_tokens):
            doc.sentences[mention.sentence_index].tokens[
                sentence_level_mention_start + i
            ] = token

    if length_difference > 0:
        for i, token in enumerate(new_tokens[: len(old_tokens)]):
            doc.sentences[mention.sentence_index].tokens[
                sentence_level_mention_start + i
            ] = token
        for i, new_token in enumerate(new_tokens[len(old_tokens) :]):
            create_token(
                doc,
                new_token,
                sentence_level_mention_start + len(old_tokens) + i,
                mention_index,
            )


# Author: Leonie
# tokens gets inserted in tokens, all index_in_doc get adjusted
def insert_token_in_tokens(
    doc: Document, token: model.Token, index_in_sentence: int
):  # passed
    if 0 <= token.sentence_index < len(doc.sentences) and 0 <= index_in_sentence <= len(
        doc.sentences[token.sentence_index].tokens
    ):
        sentence_index = token.sentence_index
        doc.sentences[sentence_index].tokens.insert(index_in_sentence, token)
        for j in range(
            index_in_sentence + 1, len(doc.sentences[sentence_index].tokens)
        ):
            doc.sentences[sentence_index].tokens[j].index_in_document += 1
        for i in range(sentence_index + 1, len(doc.sentences)):
            for tok in doc.sentences[i].tokens:
                tok.index_in_document += 1
        for mention in doc.mentions:
            if mention.sentence_index == sentence_index:
                for i in range(len(mention.token_indices)):
                    if mention.token_indices[i] >= index_in_sentence:
                        mention.token_indices[i] += 1


# Author: Leonie
# inserts token in given mention
def insert_token_in_mentions(
    doc: Document, index_in_sentence: int, mention_id
):  # passed
    try:
        if index_in_sentence not in doc.mentions[mention_id].token_indices:
            doc.mentions[mention_id].token_indices.append(index_in_sentence)
            doc.mentions[mention_id].token_indices.sort()
    except Exception as inst:
        print(type(inst))


# Author: Benedikt
# add Sentence, never used, due to the fact, that we cant create new relations etc., contains errors, not tested
def add_sentence(
    doc: Document,
    text: typing.List,
    sentence_index,
    bio_tags: typing.List = None,
    pos_tags: typing.List = None,
):
    tokens = []
    sentence = model.Sentence(tokens)
    counter = 0
    index_in_document = (
        doc.sentences[sentence - 1]
        .tokens[len(doc.senteces[sentence - 1].tokens)]
        .index_in_document
        + 1
    )
    for word in text:
        token = model.Token(
            text=word,
            index_in_document=index_in_document + counter,
            sentence_index=sentence_index,
        )
        create_token(doc, token, counter)
        counter += 1
        tokens.append(token)
    if bio_tags is not None:
        for i in range(len(tokens)):
            tokens[i].bio_tag = bio_tags[i]
    if pos_tags is not None:
        for i in range(len(tokens)):
            tokens[i].pos_tag = pos_tags[i]
    for i in range(sentence_index + 1, len(doc.sentences)):
        for token in doc.sentences[i].tokens:
            token.sentence_index += 1
    for mention in doc.mentions:
        if mention.sentence_index >= sentence_index:
            mention.sentence_index += 1
    doc.sentences.insert(sentence_index, tokens)


# Author: Benedikt
# delete a sentence from a document, used for the filters
def delete_sentence(doc: Document, sent_index: int):
    # delete every token of the sentence
    sentence = doc.sentences[sent_index]
    first_index_in_doc = sentence.tokens[0].index_in_document
    for i in range(len(sentence.tokens)):
        delete_token(doc, first_index_in_doc)

    # delete the sentence from the document.sentences
    del doc.sentences[sent_index]

    # change the sentence_index of every token of the following sentences
    for i in range(sent_index, len(doc.sentences)):
        for token in doc.sentences[i].tokens:
            token.sentence_index -= 1

    # change the sentence index of every mention with sentence_index > sent_index
    for mention in doc.mentions:
        if mention.sentence_index > sent_index:
            mention.sentence_index -= 1


# Methods needed inside the transformations #
# Author: Leonie
# get the pos Tag from the wordnet
def get_pos_tag(text: typing.List):  # text has to be an array of strings # passed
    tagged_text = nltk.pos_tag(text)
    tags = [tagged_text[i][1] for i in range(len(text))]
    return tags


# Author: Leonie
# get the bio tag based on the bio tag from the left token
def get_bio_tag_based_on_left_token(tag: str):
    if tag is "O":
        return "O"
    bio = get_bio_tag_short(tag)
    return "I-" + bio[1]


# Author: Benedikt
# returns the Bio Tag without the B or I in front
def get_bio_tag_short(tag: str):
    bio = tag.split("-")
    if len(bio) > 1:
        bio_tag = bio[1]
    else:
        bio_tag = bio[0]
    return bio_tag


# Author: Leonie
# returns the index in a sentence of a token, if the token is found in the sentence, otherwise returns None
def get_index_in_sentence(
    sent: model.Sentence, text: typing.List, index_in_doc: int = None
):
    word_count = 0
    ind_in_sent = None
    if sent == [] or text == []:
        return None
    for i in range(len(sent.tokens)):
        if index_in_doc == None:
            if sent.tokens[i].text == text[word_count]:
                word_count += 1
                if word_count == 1:
                    ind_in_sent = i
                if word_count == len(text):
                    break
            else:
                word_count = 0
        elif (
            sent.tokens[i].text == text[word_count]
            and sent.tokens[i].index_in_document == index_in_doc + word_count
        ):
            word_count += 1
            if word_count == 1:
                ind_in_sent = i
            if word_count == len(text):
                break
        else:
            word_count = 0
    if word_count == len(text):  # and ind_in_sent != None:
        return ind_in_sent
    else:
        return None


# Author: Leonie
# get the mentions from a token
def get_mentions(doc: Document, ind_in_sent, sentence_idx):
    mentions = []
    for i in range(len(doc.mentions)):
        if doc.mentions[i].sentence_index == sentence_idx:
            for j in doc.mentions[i].token_indices:
                if j == ind_in_sent:
                    mentions.append(i)
                break
    return mentions
