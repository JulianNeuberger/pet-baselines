import copy
from augment import trafo86
from augment.trafo86 import Trafo86Step
from data import model
import mockito

# Author for entire script: Leonie
def test_do_augment():
    # ARRANGE
    # Trafo Objects for testing do_augment()
    trafo1 = Trafo86Step(1, 1, False)
    trafo2 = Trafo86Step(2, 0, False)
    trafo3 = Trafo86Step(2, 0, True, 1)
    trafo4 = Trafo86Step(1, 2, True, 1)

    # test1 (Parameters: max_noun = 1, kind_of_replace = 1, no_dupl = False)
    tokens = []
    tokens.append(model.Token(text="I", index_in_document=0, pos_tag="PRP", bio_tag="", sentence_index=0))
    tokens.append(model.Token(text="am", index_in_document=1, pos_tag="VBP", bio_tag="", sentence_index=0))
    tokens.append(model.Token(text="the", index_in_document=2, pos_tag="DT", bio_tag="", sentence_index=0))
    tokens.append(model.Token(text="Head", index_in_document=3, pos_tag="NN", bio_tag="B-Actor", sentence_index=0))
    tokens.append(model.Token(text="of", index_in_document=4, pos_tag="IN", bio_tag="", sentence_index=0))
    tokens.append(model.Token(text="the", index_in_document=5, pos_tag="DT", bio_tag="", sentence_index=0))
    tokens.append(model.Token(text="functional", index_in_document=6, pos_tag="JJ", bio_tag="", sentence_index=0))
    tokens.append(model.Token(text="department", index_in_document=7, pos_tag="NN", bio_tag="B-Actor", sentence_index=0))
    tokens.append(model.Token(text=".", index_in_document=8, pos_tag=".", bio_tag="", sentence_index=0))

    sentence1 = model.Sentence(tokens=tokens)
    sentence2 = copy.deepcopy(sentence1)

    for i in range(len(sentence2.tokens)):
        sentence2.tokens[i].sentence_index = 1
        sentence2.tokens[i].index_in_document = 9 + i

    mention1 = model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[3])
    mention2 = model.Mention(ner_tag="Further Specification", sentence_index=0, token_indices=[4, 5])
    mention3 = model.Mention(ner_tag="Further Specification", sentence_index=1, token_indices=[7])
    doc = model.Document(
        text="I am the Head of the functional department.I am the Head of the functional department.",
        name="1", sentences=[sentence1, sentence2],
        mentions=[mention1, mention2, mention3],
        entities=[],
        relations=[])

    doc_to_aug1 = copy.deepcopy(doc)
    doc_sol1 = copy.deepcopy(doc)

    doc_sol1.sentences[0].tokens[3].text = "external"
    doc_sol1.sentences[0].tokens[3].pos_tag = "JJ"

    tok1 = model.Token(text="body", index_in_document=4, pos_tag="NN", bio_tag="I-Actor", sentence_index=0)
    tok2 = model.Token(text="part", index_in_document=5, pos_tag="NN", bio_tag="I-Actor", sentence_index=0)
    doc_sol1.sentences[0].tokens.insert(4, tok1)
    doc_sol1.sentences[0].tokens.insert(5, tok2)

    doc_sol1.sentences[0].tokens[6].index_in_document = 6
    doc_sol1.sentences[0].tokens[7].index_in_document = 7
    doc_sol1.sentences[0].tokens[8].index_in_document = 8
    doc_sol1.sentences[0].tokens[9].index_in_document = 9
    doc_sol1.sentences[0].tokens[10].index_in_document = 10

    doc_sol1.mentions[0].token_indices.append(4)
    doc_sol1.mentions[0].token_indices.append(5)
    doc_sol1.mentions[1].token_indices = [6, 7]
    doc_sol1.mentions[2].token_indices = [9]

    doc_sol1.sentences[1].tokens[3].text = "external"
    doc_sol1.sentences[1].tokens[3].pos_tag = "JJ"

    tok3 = model.Token(text="body", index_in_document=4,
                       pos_tag="NN", bio_tag="I-Actor",
                       sentence_index=1)
    tok4 = model.Token(text="part", index_in_document=5,
                       pos_tag="NN", bio_tag="I-Actor",
                       sentence_index=1)
    doc_sol1.sentences[1].tokens.insert(4, tok3)
    doc_sol1.sentences[1].tokens.insert(5, tok4)

    for i in range(len(doc_sol1.sentences[1].tokens)):
        doc_sol1.sentences[1].tokens[i].index_in_document = 11 + i


    # test2 (Parameters: max_noun = 2, kind_of_replace = 0, no_dupls = False)
    doc2 = copy.deepcopy(doc)
    doc_to_aug2 = copy.deepcopy(doc)

    doc_sol2 = copy.deepcopy(doc)
    doc_sol2.sentences[0].tokens[3].text = "human"
    tok5 = model.Token(text="head", index_in_document=4,
                       pos_tag="NN", bio_tag="I-Actor",
                       sentence_index=0)
    doc_sol2.sentences[0].tokens.insert(4, tok5)

    doc_sol2.sentences[0].tokens[5].index_in_document = 5
    doc_sol2.sentences[0].tokens[6].index_in_document = 6
    doc_sol2.sentences[0].tokens[7].index_in_document = 7

    tok6 = model.Token(text="academic", index_in_document=8,
                       pos_tag="JJ", bio_tag="B-Actor",
                       sentence_index=0)
    doc_sol2.sentences[0].tokens.insert(8, tok6)

    doc_sol2.sentences[0].tokens[9].index_in_document = 9
    doc_sol2.sentences[0].tokens[9].bio_tag = "I-Actor"
    doc_sol2.sentences[0].tokens[10].index_in_document = 10

    sent2 = copy.deepcopy(doc_sol2.sentences[0])
    doc_sol2.sentences[1] = sent2

    for i in range(len(doc_sol2.sentences[1].tokens)):
        doc_sol2.sentences[1].tokens[i].sentence_index = 1
        doc_sol2.sentences[1].tokens[i].index_in_document = 11 + i

    doc_sol2.mentions[0].token_indices.append(4)
    doc_sol2.mentions[1].token_indices = [5, 6]
    doc_sol2.mentions[2].token_indices = [8, 9]


    # test3 (Parameters: max_noun = 2, kind_of_replace = 0, no_dupl = True)
    doc3 = copy.deepcopy(doc)
    doc3.sentences[0].tokens[7].text = "Head"
    doc3.sentences[0].tokens[7].pos_tag = "NN"
    doc3.sentences[1].tokens[7].text = "Head"
    doc3.sentences[1].tokens[7].pos_tag = "NN"

    doc_to_aug3 = copy.deepcopy(doc3)

    doc_sol3 = copy.deepcopy(doc3)
    doc_sol3.sentences[0].tokens[3].text = "human"
    tok7 = model.Token(text="head", index_in_document=4,
                       pos_tag="NN", bio_tag="I-Actor",
                       sentence_index=0)
    doc_sol3.sentences[0].tokens.insert(4, tok7)
    for i in range(5, 10):
        doc_sol3.sentences[0].tokens[i].index_in_document += 1

    doc_sol3.sentences[1].tokens[0].index_in_document = 10
    doc_sol3.sentences[1].tokens[1].index_in_document = 11
    doc_sol3.sentences[1].tokens[2].index_in_document = 12
    doc_sol3.sentences[1].tokens[3].text = "human"
    doc_sol3.sentences[1].tokens[3].index_in_document = 13
    tok7 = model.Token(text="head", index_in_document=14,
                       pos_tag="NN", bio_tag="I-Actor",
                       sentence_index=1)
    doc_sol3.sentences[1].tokens.insert(4, tok7)
    for i in range(5, 10):
        doc_sol3.sentences[1].tokens[i].index_in_document = 10 + i

    doc_sol3.mentions[0].token_indices.append(4)
    doc_sol3.mentions[1].token_indices = [5, 6]
    doc_sol3.mentions[2].token_indices = [8]


    # test4: (parameters: max_noun = 1, kind of replace = 2, no dupl = True)
    doc4 = copy.deepcopy(doc)
    doc_sol4 = copy.deepcopy(doc_sol1)
    doc_to_aug4 = copy.deepcopy(doc4)


    # ACT
    # test1
    token_list1 = [model.Token(text='Head', index_in_document=3, pos_tag='NN', bio_tag='B-Actor', sentence_index=0), model.Token(text='department', index_in_document=7, pos_tag='NN', bio_tag='B-Actor', sentence_index=0)]
    token_list2 = [model.Token(text='Head', index_in_document=14, pos_tag='NN', bio_tag='B-Actor', sentence_index=1), model.Token(text='department', index_in_document=18, pos_tag='NN', bio_tag='B-Actor', sentence_index=1)]
    mockito.when(trafo86).rand().thenReturn(0.2)
    mockito.when(trafo86).shuffle(token_list1).thenReturn(token_list1)
    mockito.when(trafo86).shuffle(token_list2).thenReturn(token_list2)
    doc_aug1 = trafo1.do_augment(doc_to_aug1)

    # test2
    token_list3 = [model.Token(text='department', index_in_document=7, pos_tag='NN', bio_tag='B-Actor', sentence_index=0), model.Token(text='Head', index_in_document=3, pos_tag='NN', bio_tag='B-Actor', sentence_index=0)]
    token_list4 = [model.Token(text='department', index_in_document=18, pos_tag='NN', bio_tag='B-Actor', sentence_index=1), model.Token(text='Head', index_in_document=14, pos_tag='NN', bio_tag='B-Actor', sentence_index=1)]
    mockito.when(trafo86).rand().thenReturn(0.2).thenReturn(0.2)
    mockito.when(trafo86).shuffle(token_list1).thenReturn(token_list3)
    mockito.when(trafo86).shuffle(token_list2).thenReturn(token_list4)
    doc_aug2 = trafo2.do_augment(doc_to_aug2)

    # test3
    token_list5 = [model.Token(text='Head', index_in_document=3, pos_tag='NN', bio_tag='B-Actor', sentence_index=0)]
    token_list6 = [model.Token(text='Head', index_in_document=13, pos_tag='NN', bio_tag='B-Actor', sentence_index=1)]
    mockito.when(trafo86).shuffle(token_list5).thenReturn(token_list5)
    mockito.when(trafo86).shuffle(token_list6).thenReturn(token_list6)
    doc_aug3 = trafo3.do_augment(doc_to_aug3)

    # test4
    mockito.when(trafo86).rand().thenReturn(0.7).thenReturn(0.7)
    mockito.when(trafo86).shuffle(token_list1).thenReturn(token_list1)
    mockito.when(trafo86).shuffle(token_list2).thenReturn(token_list2)
    doc_aug4 = trafo4.do_augment(doc_to_aug4)



    # ASSERT
    assert doc_aug1 == doc_sol1  # test1
    assert doc_aug2 == doc_sol2  # test2
    assert doc_aug3 == doc_sol3  # test3
    assert doc_aug4 == doc_sol4  # test4
