from data import model, Document
import typing
import data

docs: typing.List[model.Document] = data.loader.read_documents_from_json('../one_doc.json')

def delete_Token(index_in_document):
    for doc in docs:
        index = delete_token_from_tokens(doc, index_in_document)
        index_in_sentence = index[0]
        sentence_index = index[1]
        mention_index = delete_token_from_mention_token_indices(doc, index_in_sentence, sentence_index)
        if mention_index != None:
            delete_mention_from_entity(doc, mention_index)
        break

def delete_token_from_tokens(doc: Document, index_in_document: int) -> typing.List:
    not_found_token = True
    index = []
    for sentence in doc.sentences:
        counter = 0
        for token in sentence.tokens:
            if token.index_in_document == index_in_document and not_found_token:
                not_found_token = False
                del sentence.tokens[counter]
                sentence.tokens[counter].index_in_document -= 1
                index.append(counter)
                index.append(token.sentence_index)
            if not not_found_token:
                token.index_in_document -= 1
            counter += 1
    return index


def delete_token_from_mention_token_indices(doc: Document, index_in_sentence, sentence_index) -> int:
    counter = 0
    for mention in doc.mentions:
        if mention.sentence_index == sentence_index:
            if index_in_sentence in mention.token_indices:
                if len(mention.token_indices) == 1:
                    doc.mentions.remove(mention)
                    change_mention_indices_in_entities(doc, counter)
                    return counter
                else:
                    mention.token_indices.remove(index_in_sentence)
                    for i in range(len(mention.token_indices)):
                        if mention.token_indices[i] > index_in_sentence:
                            mention.token_indices[i] -= 1
                    return None
        counter += 1
    print(doc.mentions)


def change_mention_indices_in_entities(doc: Document, mention_index: int):
    for entity in doc.entities:
        for i in range(len(entity.mention_indices)):
            if entity.mention_indices[i] > mention_index:
                entity.mention_indices[i] -= 1



def delete_mention_from_entity(doc: Document, mention_index: int):
    counter = 0
    for entity in doc.entities:
        if mention_index in entity.mention_indices:
            if len(entity.mention_indices) == 1:
                doc.entities.remove(entity)
                delete_relations(doc, counter)
                break
            else:
                entity.mention_indices.remove(mention_index)
        counter += 1


def delete_relations(doc: Document, entity_index: int):
    for relation in doc.relations:
        if entity_index in [relation.head_entity_index, relation.tail_entity_index]:
            doc.relations.remove(relation)




delete_Token(14)