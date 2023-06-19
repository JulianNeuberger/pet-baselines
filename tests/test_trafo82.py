import copy
from augment.trafo82 import Trafo82Step
from data import model
import mockito
from augment import trafo82


def test_separate_into_contracted_and_expanded_form():
    # Arrange
    trafo1 = Trafo82Step(False, False, False)
    str1 = [["ACCT", "account"]]
    sol1 = ["ACCT"]
    sol2 = ["account"]
    sol3 = (sol1, sol2)

    # Act
    act1 = trafo1.separate_into_contracted_and_expanded_form(str1, True)

    # Assert
    assert act1 == sol3


def test_do_augment():
    # Arrange
    trafo1 = Trafo82Step(True, False, 1)
    trafo2 = Trafo82Step(False, True, 1)
    trafo3 = Trafo82Step(True, True, 1)
    trafo4 = Trafo82Step(True, True, 2)
    tokens = [model.Token(text="I", index_in_document=0,
                          pos_tag="PRP", bio_tag="O",
                          sentence_index=0), model.Token(text="leave", index_in_document=1,
                                                         pos_tag="VBP", bio_tag="O",
                                                         sentence_index=0), model.Token(text="HR", index_in_document=2,
                                                                                        pos_tag="NN", bio_tag="O",
                                                                                        sentence_index=0),
              model.Token(text="office", index_in_document=3,
                          pos_tag="NN", bio_tag="O",
                          sentence_index=0)]

    sentence1 = model.Sentence(tokens=tokens)

    doc = model.Document(
        text="I leave HR office",
        name="1", sentences=[sentence1],
        mentions=[],
        entities=[],
        relations=[])

    tokens2 = [model.Token(text="HR", index_in_document=0,
                           pos_tag="NN", bio_tag="O",
                           sentence_index=0), model.Token(text="Human", index_in_document=1,
                                                          pos_tag="NNP", bio_tag="O",
                                                          sentence_index=0),
               model.Token(text="Resources", index_in_document=2,
                           pos_tag="NNS", bio_tag="O",
                           sentence_index=0)]
    sentence2 = model.Sentence(tokens=tokens2)

    # Doc 3 for both cases
    doc3 = model.Document(
        text="HR Human resources",
        name="1", sentences=[sentence2],
        mentions=[],
        entities=[],
        relations=[])

    doc_to_aug1 = copy.deepcopy(doc)
    doc_to_aug3 = copy.deepcopy(doc3)

    doc_sol = copy.deepcopy(doc)
    doc_sol.sentences[0].tokens[2].text = "Human"
    doc_sol.sentences[0].tokens[2].pos_tag = "NNP"
    doc_sol.sentences[0].tokens.insert(3, model.Token(text="Resources", index_in_document=3,
                                                      pos_tag="NNS", bio_tag="O",
                                                      sentence_index=0))
    doc_sol.sentences[0].tokens[4].index_in_document = 4

    doc_to_aug2 = copy.deepcopy(doc_sol)
    doc_sol2 = copy.deepcopy(doc)
    tokens3 = [model.Token(text="Human", index_in_document=0,
                           pos_tag="NNP", bio_tag="O",
                           sentence_index=0), model.Token(text="Resources", index_in_document=1,
                                                          pos_tag="NNS", bio_tag="O",
                                                          sentence_index=0), model.Token(text="HR", index_in_document=2,
                                                                                         pos_tag="NN", bio_tag="O",
                                                                                         sentence_index=0)]
    sentence3 = model.Sentence(tokens=tokens3)
    doc_sol3 = model.Document(
        text="HR Human resources",
        name="1", sentences=[sentence3],
        mentions=[],
        entities=[],
        relations=[])

    # Doc 4 for trafo 27 bank
    tokens3 = [model.Token(text="I'm", index_in_document=0,
                           pos_tag="NN", bio_tag="O",
                           sentence_index=0), model.Token(text="I", index_in_document=1,
                                                          pos_tag="NNP", bio_tag="O",
                                                          sentence_index=0),
               model.Token(text="am", index_in_document=2,
                           pos_tag="NNS", bio_tag="O",
                           sentence_index=0)]
    sentence3 = model.Sentence(tokens=tokens3)
    doc4 = model.Document(
        text="Im i am",
        name="1", sentences=[sentence3],
        mentions=[],
        entities=[],
        relations=[])

    doc_sol4 = copy.deepcopy(doc4)
    doc_sol4.sentences[0].tokens[0].text = "I"
    doc_sol4.sentences[0].tokens[0].pos_tag = "PRP"
    doc_sol4.sentences[0].tokens[1].text = "am"
    doc_sol4.sentences[0].tokens[1].pos_tag = "VBP"
    doc_sol4.sentences[0].tokens[2].text = "I'm"
    doc_sol4.sentences[0].tokens[2].pos_tag = "NN"
    doc_to_aug4 = copy.deepcopy(doc4)
    # Act
    mockito.when(trafo82).randint(0, 2).thenReturn(1)
    mockito.when(trafo82).randint(0, 0).thenReturn(0)
    doc_aug1 = trafo1.do_augment(doc_to_aug1)
    doc_aug2 = trafo2.do_augment(doc_to_aug2)
    doc_aug3 = trafo3.do_augment(doc_to_aug3)
    doc_aug4 = trafo4.do_augment(doc_to_aug4)


    # Assert

    assert doc_aug1 == doc_sol
    assert doc_aug2 == doc_sol2
    assert doc_aug3 == doc_sol3
    assert doc_aug4 == doc_sol4