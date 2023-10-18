import copy
from data import model
from augment.trafo5 import Trafo5Step

# Author for entire script: Benedikt
def test_is_ant():
    # Arrange
    trafo = Trafo5Step()
    word1 = "evil"
    word2 = "good"
    word3 = "good"
    word4 = "nice"
    sol1 = True
    sol2 = False

    # Act
    is_ant1 = trafo.is_ant(word1=word1, word2=word2)
    is_ant2 = trafo.is_ant(word1=word3, word2=word4)
    # Assert
    assert is_ant1 == sol1
    assert is_ant2 == sol2


def test_is_syn():
    # Arrange
    trafo = Trafo5Step()
    word1 = "evil"
    word2 = "good"
    word3 = "people"
    word4 = "masses"
    sol1 = False
    sol2 = True

    # Act
    is_ant1 = trafo.is_syn(word1=word1, word2=word2)
    is_ant2 = trafo.is_syn(word1=word3, word2=word4)

    # Assert
    assert is_ant1 == sol1
    assert is_ant2 == sol2


def test_is_ant_syn():
    # Arrange
    trafo = Trafo5Step()
    token1 = model.Token(text="evil", index_in_document=1, pos_tag="test", bio_tag="test", sentence_index=1)
    token2 = model.Token(text="good", index_in_document=1, pos_tag="test", bio_tag="test", sentence_index=1)
    token3 = model.Token(text="people", index_in_document=1, pos_tag="test", bio_tag="test", sentence_index=1)
    token4 = model.Token(text="persons", index_in_document=1, pos_tag="test", bio_tag="test", sentence_index=1)

    token5 = model.Token(text="calm", index_in_document=1, pos_tag="test", bio_tag="test", sentence_index=1)
    token6 = model.Token(text="good", index_in_document=1, pos_tag="test", bio_tag="test", sentence_index=1)
    token7 = model.Token(text="people", index_in_document=1, pos_tag="test", bio_tag="test", sentence_index=1)
    token8 = model.Token(text="persons", index_in_document=1, pos_tag="test", bio_tag="test", sentence_index=1)

    token9 = model.Token(text="calm", index_in_document=1, pos_tag="test", bio_tag="test", sentence_index=1)
    token10 = model.Token(text="good", index_in_document=1, pos_tag="test", bio_tag="test", sentence_index=1)
    token11 = model.Token(text="people", index_in_document=1, pos_tag="test", bio_tag="test", sentence_index=1)
    token12 = model.Token(text="masses", index_in_document=1, pos_tag="test", bio_tag="test", sentence_index=1)

    token_list1 = []
    token_list1.append(token1)
    token_list1.append(token2)
    token_list1.append(token3)
    token_list1.append(token4)

    token_list2 = []
    token_list2.append(token5)
    token_list2.append(token6)
    token_list2.append(token7)
    token_list2.append(token8)

    token_list3 = []
    token_list3.append(token9)
    token_list3.append(token10)
    token_list3.append(token11)
    token_list3.append(token12)

    sol1 = True
    sol2 = False
    sol3 = True
    # Act
    is_ant_syn1 = trafo.is_ant_syn(token_list1)
    is_ant_syn2 = trafo.is_ant_syn(token_list2)
    is_ant_syn3 = trafo.is_ant_syn(token_list3)
    #Assert
    assert is_ant_syn1 == sol1
    assert is_ant_syn2 == sol2
    assert is_ant_syn3 == sol3


def test_do_augment():
    # Arrange
    trafo = Trafo5Step()
    tokens = []
    tokens.append(model.Token(text="I", index_in_document=0,
                              pos_tag="PRP", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="am", index_in_document=1,
                              pos_tag="VBP", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="good", index_in_document=2,
                              pos_tag="JJ", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="worker", index_in_document=3,
                              pos_tag="NN", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="of", index_in_document=4,
                              pos_tag="IN", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="the", index_in_document=5,
                              pos_tag="DT", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="functional", index_in_document=6,
                              pos_tag="JJ", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="department", index_in_document=7,
                              pos_tag="NN", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text=".", index_in_document=8,
                              pos_tag=".", bio_tag="",
                              sentence_index=0))
    sentence1 = model.Sentence(tokens=tokens)

    doc = model.Document(
        text="I am the Head of the functional department.I am the Head of the functional department available.",
        name="1", sentences=[sentence1],
        mentions=[],
        entities=[],
        relations=[])

    doc2 = copy.deepcopy(doc)
    doc2.sentences[0].tokens.pop(2)

    doc3 = copy.deepcopy(doc)
    doc3.sentences[0].tokens[6].text = "evil"

    doc_sol1 = copy.deepcopy(doc)
    doc_sol1.sentences[0].tokens[2].text = "bad"
    doc_sol1.sentences[0].tokens[6].text = "nonfunctional"

    doc_sol2 = copy.deepcopy(doc2)

    doc_sol3 = copy.deepcopy(doc3)
    # Act

    doc_aug1 = trafo.do_augment(doc)
    doc_aug2 = trafo.do_augment(doc2)
    doc_aug3 = trafo.do_augment(doc3)
    #Assert

    assert doc_aug1 == doc_sol1
    assert doc_aug2 == doc_sol2
    assert doc_aug3 == doc_sol3