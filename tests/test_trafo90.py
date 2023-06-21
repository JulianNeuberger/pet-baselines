import copy
from augment import trafo90
from augment.trafo90 import Trafo90Step
from data import model
import mockito

#  Shuffle within Segments


def test_do_augment():
    # ARRANGE
    # Trafo Object for testing do_augment()
    trafo1 = Trafo90Step()

    # test1
    tokens = []
    tokens.append(model.Token(text="I", index_in_document=0, pos_tag="PRP", bio_tag="B-Actor", sentence_index=0))
    tokens.append(model.Token(text="am", index_in_document=1, pos_tag="VBP", bio_tag="I-Actor", sentence_index=0))
    tokens.append(model.Token(text="the", index_in_document=2, pos_tag="DT", bio_tag="O", sentence_index=0))
    tokens.append(model.Token(text="Head", index_in_document=3, pos_tag="NN", bio_tag="B-Activity", sentence_index=0))
    tokens.append(model.Token(text="of", index_in_document=4, pos_tag="IN", bio_tag="I-Activity", sentence_index=0))
    tokens.append(model.Token(text="the", index_in_document=5, pos_tag="DT", bio_tag="O", sentence_index=0))
    tokens.append(model.Token(text="functional", index_in_document=6, pos_tag="JJ", bio_tag="O", sentence_index=0))
    tokens.append(model.Token(text="department", index_in_document=7, pos_tag="NN", bio_tag="O", sentence_index=0))
    tokens.append(model.Token(text=".", index_in_document=8, pos_tag=".", bio_tag="O", sentence_index=0))

    sentence1 = model.Sentence(tokens)
    sentence2 = copy.deepcopy(sentence1)
    sentence2.tokens[5].bio_tag = "B-Actor"
    sentence2.tokens[6].bio_tag = "I-Actor"
    sentence2.tokens[7].bio_tag = "B-Activity"

    for i in range(9):
        sentence2.tokens[i].sentence_index = 1

    doc = model.Document(
        text="I am the Head of the functional department.I am the Head of the functional department.",
        name="1", sentences=[sentence1, sentence2],
        mentions=[],
        entities=[],
        relations=[])

    doc_sol1 = copy.deepcopy(doc)

    tok0 = copy.deepcopy(doc_sol1.sentences[0].tokens[1])
    tok1 = copy.deepcopy(doc_sol1.sentences[0].tokens[0])
    doc_sol1.sentences[0].tokens[0] = tok0
    doc_sol1.sentences[0].tokens[1] = tok1
    doc_sol1.sentences[0].tokens[0].bio_tag = "B-Actor"
    doc_sol1.sentences[0].tokens[1].bio_tag = "I-Actor"

    tok5 = copy.deepcopy(doc_sol1.sentences[0].tokens[6])
    tok6 = copy.deepcopy(doc_sol1.sentences[0].tokens[5])
    tok7 = copy.deepcopy(doc_sol1.sentences[0].tokens[8])
    tok8 = copy.deepcopy(doc_sol1.sentences[0].tokens[7])
    doc_sol1.sentences[0].tokens[5] = tok5
    doc_sol1.sentences[0].tokens[6] = tok6
    doc_sol1.sentences[0].tokens[7] = tok7
    doc_sol1.sentences[0].tokens[8] = tok8

    for i in range(9):
        doc_sol1.sentences[0].tokens[i].index_in_document = i

    tok_3 = copy.deepcopy(doc_sol1.sentences[1].tokens[4])
    tok_4 = copy.deepcopy(doc_sol1.sentences[1].tokens[3])
    doc_sol1.sentences[1].tokens[3] = tok_3
    doc_sol1.sentences[1].tokens[4] = tok_4
    doc_sol1.sentences[1].tokens[3].bio_tag = "B-Activity"
    doc_sol1.sentences[1].tokens[4].bio_tag = "I-Activity"

    tok_5 = copy.deepcopy(doc_sol1.sentences[1].tokens[6])
    tok_6 = copy.deepcopy(doc_sol1.sentences[1].tokens[5])
    doc_sol1.sentences[1].tokens[5] = tok_5
    doc_sol1.sentences[1].tokens[6] = tok_6
    doc_sol1.sentences[1].tokens[5].bio_tag = "B-Actor"
    doc_sol1.sentences[1].tokens[6].bio_tag = "I-Actor"

    for i in range(9):
        doc_sol1.sentences[1].tokens[i].index_in_document = i

    # ACT
    # test1
    mockito.when(trafo90).binomial(1, 0.5).thenReturn(True).thenReturn(False).thenReturn(False).thenReturn(True).thenReturn(False).thenReturn(False).thenReturn(True).thenReturn(True).thenReturn(False).thenReturn(True)

    mockito.when(trafo90).shuffle([0,1]).thenReturn([1, 0])
    mockito.when(trafo90).shuffle([5, 6, 7, 8]).thenReturn([6, 5, 8, 7])
    mockito.when(trafo90).shuffle([3, 4]).thenReturn([4, 3])
    mockito.when(trafo90).shuffle([5, 6]).thenReturn([6, 5])
    mockito.when(trafo90).shuffle([8]).thenReturn([8])

    doc_aug = trafo1.do_augment(doc)

    # ASSERT
    assert doc_aug == doc_sol1
