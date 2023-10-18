import copy
from augment.trafo100 import Trafo100Step
from data import model
import mockito
from augment import trafo100

# Author for entire script: Benedikt
def test_do_augment():
    # Arrange
    trafo = Trafo100Step()
    tokens = [model.Token(text="good", index_in_document=0,
                          pos_tag="JJ", bio_tag="O",
                          sentence_index=0),
              model.Token(text="leave", index_in_document=1,
                          pos_tag="VBP", bio_tag="O",
                          sentence_index=0),
              model.Token(text="head", index_in_document=2,
                          pos_tag="NN", bio_tag="O",
                          sentence_index=0),
              model.Token(text=".", index_in_document=3,
                          pos_tag=".", bio_tag="O",
                          sentence_index=0)]
    sentence1 = model.Sentence(tokens=tokens)

    doc = model.Document(
        text="good leave head .",
        name="1", sentences=[sentence1],
        mentions=[],
        entities=[],
        relations=[])

    doc_sol1 = copy.deepcopy(doc)
    tok1 = model.Token(text="full", index_in_document=1,
                       pos_tag="JJ", bio_tag="O",
                       sentence_index=0)
    tok2 = model.Token(text="exit", index_in_document=3,
                       pos_tag="NN", bio_tag="O",
                       sentence_index=0)
    tok3 = model.Token(text="mind", index_in_document=5,
                       pos_tag="NN", bio_tag="O",
                       sentence_index=0)
    doc_sol1.sentences[0].tokens.insert(1, tok1)
    doc_sol1.sentences[0].tokens.insert(3, tok2)
    doc_sol1.sentences[0].tokens.insert(5, tok3)
    doc_sol1.sentences[0].tokens[2].index_in_document = 2
    doc_sol1.sentences[0].tokens[4].index_in_document = 4
    doc_sol1.sentences[0].tokens[6].index_in_document = 6

    doc_to_aug = copy.deepcopy(doc)
    # Act
    doc_aug1 = trafo.do_augment(doc_to_aug)


    # Assert
    assert doc_aug1 == doc_sol1