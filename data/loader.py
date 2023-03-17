import json
import typing

import nltk

from data import model

nltk.download('averaged_perceptron_tagger')


def read_names(filename) -> typing.List[typing.List[str]]:
    data = open(filename).readlines()
    data = [item.strip().split('\t') for item in data]
    return data


def read_documents_from_json(file_path: str) -> typing.List[model.Document]:
    documents = []
    with open(file_path, 'r', encoding='utf8') as f:
        for json_line in f:
            json_data = json.loads(json_line)
            documents.append(_read_document_from_json(json_data))

    return documents


def _read_document_from_json(json_data: typing.Dict) -> model.Document:
    mentions = _read_mentions_from_json(json_data['entities'])
    relations = _read_relations_from_json(json_data['relations'], mentions)
    return model.Document(
        text=json_data['text'],
        sentences=_read_sentences_from_json(json_data['tokens']),
        mentions=mentions,
        relations=relations
    )


def _read_sentences_from_json(json_tokens: typing.List[typing.Dict]) -> typing.List[model.Sentence]:
    tokens = []
    for i, json_token in enumerate(json_tokens):
        tokens.append((json_token['sentence_id'], model.Token(
            text=json_token['text'],
            pos_tag=json_token['stanza_pos'],
            bio_tag=json_token['ner'],
            index_in_document=i
        )))

    sentences = []
    cur_sentence_id: typing.Optional[int] = None
    for sentence_id, token in tokens:
        if sentence_id != cur_sentence_id:
            sentences.append(model.Sentence())
        sentences[-1].tokens.append(token)
        cur_sentence_id = sentence_id

    return sentences


def _read_mentions_from_json(json_mentions: typing.List[typing.Dict]) -> typing.List[model.Mention]:
    mentions = []
    for json_mention in json_mentions:
        mention = _read_mention_from_json(json_mention)
        mentions.append(mention)
    return mentions


def _read_mention_from_json(json_mention: typing.Dict) -> model.Mention:
    return model.Mention(ner_tag=json_mention['ner'],
                         sentence_index=json_mention['sentence_id'],
                         token_indices=json_mention['token_indices'])


def _read_relations_from_json(json_relations: typing.List[typing.Dict],
                              mentions: typing.List[model.Mention]) -> typing.List[model.Relation]:
    relations = []
    mentions_as_tuples = [m.to_tuple() for m in mentions]
    for json_relation in json_relations:
        head_mention = _read_mention_from_json(json_relation['head_entity'])
        tail_mention = _read_mention_from_json(json_relation['tail_entity'])

        head_index = mentions_as_tuples.index(head_mention.to_tuple())
        tail_index = mentions_as_tuples.index(tail_mention.to_tuple())

        relations.append(model.Relation(
            head_entity_index=head_index,
            tail_entity_index=tail_index,
            tag=json_relation['type']
        ))

    return relations
