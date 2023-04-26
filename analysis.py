#Analysen des Datensatzes
#pos_tags: PRP: personal pronoun
        # VBN: verb past participle
        # VBP: verb singular present
        # VBZ: verb singular present third person
        # DT: determiner
        # NN: noun
        # IN: preposition or subordination conjunction
        # NNS: noun plural,
        # NNP: proper noun singular (eigenname)
        # JJ: adjective
        # RB: adverb
        # WP: Wh-pronoun
        #PRP$: possesive pronoun


from data import model
import typing
import data

#Datensatz als Liste
docs: typing.List[model.Document]= data.loader.read_documents_from_json('./pet_dataset.json')

def count_adjectives_in_pet_dataset(): #Anzahl: 187
    counter = 0
    for doc in docs:
        for mention in doc.mentions:
            for token in mention.get_tokens(doc):
                if token.pos_tag == "JJ" or token.pos_tag == "JJR" or token.pos_tag == "JJS":
                    print(token.text)
                    counter += 1
    print(counter)

def count_adverbes_in_pet_dataset(): #Anzahl: 63
    counter = 0
    for doc in docs:
        for mention in doc.mentions:
            for token in mention.get_tokens(doc):
                if token.pos_tag == "RB" or token.pos_tag == "RBR" or token.pos_tag == "RBS":
                    print(token.text)
                    counter += 1
    print(counter)

def count_numbers_in_pet_dataset(): #Anzahl: 63
    counter = 0
    counter_number = 0
    for doc in docs:
        for mention in doc.mentions:
            for token in mention.get_tokens(doc):
                if token.pos_tag == "CD":
                    counter_number += 1
                    print(token.text)
    print(counter)
    print(counter_number)


def adjectives_belonging_entities():
    counter_further = 0
    counter_cond = 0
    counter_actor = 0
    counter_activity_data = 0
    counter_xor = 0
    counter_and = 0
    counter_activity = 0
    counter = 0
    for doc in docs:
        for entity in doc.entities:
            for mentIndex in entity.mention_indices:
                mention = doc.mentions[mentIndex]
                for token in mention.get_tokens(doc):
                    if token.pos_tag == "JJ" or token.pos_tag == "JJR" or token.pos_tag == "JJS":
                        print(entity.get_tag(doc))
                        if entity.get_tag(doc) == "Further Specification":
                            counter_further += 1
                        elif entity.get_tag(doc) == "Condition Specification":
                            counter_cond += 1
                        elif entity.get_tag(doc) == "Actor":
                            counter_actor += 1
                        elif entity.get_tag(doc) == "Activity Data":
                            counter_activity_data+= 1
                        elif entity.get_tag(doc) == "XOR Gateway":
                            counter_xor += 1
                        elif entity.get_tag(doc) == "Activity":
                            counter_activity += 1
                        elif entity.get_tag(doc) == "AND Gateway":
                            counter_and += 1
                        else:
                            counter += 1
    print(f'Rest: {counter}') #113
    print(f'Further Specification: {counter_further}') #32
    print(f'Condition Specification: {counter_cond}') #42
    print(f'XOR Gateway: {counter_xor}')
    print(f'AND Gateway: {counter_and}')
    print(f'Activity Data: {counter_activity_data}')
    print(f'Activity: {counter_activity}')
    print(f'Actor: {counter_actor}')

def numbers_belonging_entities():
    counter_further = 0
    counter_cond = 0
    counter_actor = 0
    counter_activity_data = 0
    counter_xor = 0
    counter_and = 0
    counter_activity = 0
    counter = 0
    for doc in docs:
        for entity in doc.entities:
            for mentIndex in entity.mention_indices:
                mention = doc.mentions[mentIndex]
                for token in mention.get_tokens(doc):
                    if token.pos_tag == "CD" :
                        print(entity.get_tag(doc))
                        if entity.get_tag(doc) == "Further Specification":
                            counter_further += 1
                        elif entity.get_tag(doc) == "Condition Specification":
                            counter_cond += 1
                        elif entity.get_tag(doc) == "Actor":
                            counter_actor += 1
                        elif entity.get_tag(doc) == "Activity Data":
                            counter_activity_data += 1
                        elif entity.get_tag(doc) == "XOR Gateway":
                            counter_xor += 1
                        elif entity.get_tag(doc) == "Activity":
                            counter_activity += 1
                        elif entity.get_tag(doc) == "AND Gateway":
                            counter_and += 1
                        else:
                            counter += 1
    print(f'Rest: {counter}')  # 113
    print(f'Further Specification: {counter_further}')  # 32
    print(f'Condition Specification: {counter_cond}')  # 42
    print(f'XOR Gateway: {counter_xor}')
    print(f'AND Gateway: {counter_and}')
    print(f'Activity Data: {counter_activity_data}')
    print(f'Activity: {counter_activity}')
    print(f'Actor: {counter_actor}')

def nouns_belonging_entities():
    counter_further = 0
    counter_cond = 0
    counter_actor = 0
    counter_activity_data = 0
    counter_xor = 0
    counter_and = 0
    counter_activity = 0
    counter = 0
    for doc in docs:
        for entity in doc.entities:
            for mentIndex in entity.mention_indices:
                mention = doc.mentions[mentIndex]
                for token in mention.get_tokens(doc):
                    if token.pos_tag == "NN" or token.pos_tag == "NNS":
                        print(entity.get_tag(doc))
                        if entity.get_tag(doc) == "Further Specification":
                            counter_further += 1
                        elif entity.get_tag(doc) == "Condition Specification":
                            counter_cond += 1
                        elif entity.get_tag(doc) == "Actor":
                            counter_actor += 1
                        elif entity.get_tag(doc) == "Activity Data":
                            counter_activity_data+= 1
                        elif entity.get_tag(doc) == "XOR Gateway":
                            counter_xor += 1
                        elif entity.get_tag(doc) == "Activity":
                            counter_activity += 1
                        elif entity.get_tag(doc) == "AND Gateway":
                            counter_and += 1
                        else:
                            counter += 1
    print(f'Rest: {counter}') #113
    print(f'Further Specification: {counter_further}') #32
    print(f'Condition Specification: {counter_cond}') #42
    print(f'XOR Gateway: {counter_xor}')
    print(f'AND Gateway: {counter_and}')
    print(f'Activity Data: {counter_activity_data}')
    print(f'Activity: {counter_activity}')
    print(f'Actor: {counter_actor}')

def all_ner_tags():
    ner_tag_list = ["NER TAG LIST:"]
    for doc in docs:
        for mention in doc.mentions:
            tag = mention.ner_tag
            if tag in ner_tag_list:
                pass
            else:
                ner_tag_list.append(tag)
    for t in ner_tag_list:
        print(t)

def all_bio_tags():
    bio_tag_list = ["BIO TAG LIST"]
    for doc in docs:
        for mention in doc.mentions:
            for token in mention.get_tokens(doc):
                tok = token.bio_tag
                if tok in bio_tag_list:
                    pass
                else:
                    bio_tag_list.append(tok)
    for t in bio_tag_list:
        print(t)

def all_ner_tag_actors():
    for doc in docs:
        for mention in doc.mentions:
            if mention.ner_tag == "Actor":
                for token in mention.get_tokens(doc):
                    print(token.text)


def count_question_marks():
    question_mark_counter = 0
    for doc in docs:
        for mention in doc.mentions:
            for token in mention.get_tokens(doc):
                if token.text == "?" or token.text == " ? " or token.text == "? " or token.text == " ?":
                    print(token.text)
                    question_mark_counter += 1
    print(question_mark_counter)

def count_and_gateway_pos_tags():
    counter = 0
    counter_IN = 0
    counter_NNP = 0
    counter_DT = 0
    counter_NN = 0
    counter_CD = 0
    counter_JJ = 0
    counter_NNS = 0
    counter_VBN = 0
    counter_VBP = 0
    counter_VB = 0


    for doc in docs:
        for entity in doc.entities:
            if entity.get_tag(doc) == "AND Gateway":
                for mentIndex in entity.mention_indices:
                    mention = doc.mentions[mentIndex]
                    for token in mention.get_tokens(doc):
                        print(token.pos_tag)
                        counter += 1
                        if token.pos_tag == "IN":
                            counter_IN += 1
                        elif token.pos_tag == "NNP":
                            counter_NNP += 1
                        elif token.pos_tag == "DT":
                            counter_DT += 1
                        elif token.pos_tag == "NN":
                            counter_NN += 1
                        elif token.pos_tag == "CD":
                            counter_CD += 1
                        elif token.pos_tag == "JJ":
                            counter_JJ += 1
                        elif token.pos_tag == "NNS":
                            counter_NNS += 1
                        elif token.pos_tag == "VBN":
                            counter_VBN += 1
                        elif token.pos_tag == "VBP":
                            counter_VBP += 1
                        elif token.pos_tag == "VB":
                            counter_VB += 1
    print(f'Counter allgemein {counter}')
    print(f'Counter IN {counter_IN}')
    print(f'Counter NNP {counter_NNP}')
    print(f'Counter DT {counter_DT}')
    print(f'Counter NN {counter_NN}')
    print(f'Counter CD {counter_CD}')
    print(f'Counter JJ {counter_JJ}')
    print(f'Counter NNS {counter_NNS}')
    print(f'Counter VBN {counter_VBN}')
    print(f'Counter VBP {counter_VBP}')
    print(f'Counter VB {counter_VB}')
#count_adjectives_in_pet_dataset()
#count_adverbes_in_pet_dataset()
#adjectives_belonging_entities()
#count_numbers_in_pet_dataset()
#numbers_belonging_entities()
#all_ner_tags()
#all_bio_tags()
#all_ner_tag_actors()
#nouns_belonging_entities()
#count_question_marks()
#count_and_gateway_pos_tags()