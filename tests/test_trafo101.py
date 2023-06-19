import copy

import pytest
from augment.trafo101 import Trafo101Step
from data import model
import mock
def test_do_augment():
    # Arrange
    trafo = Trafo101Step(type=True, no_dupl=False)
    trafo2 = Trafo101Step(type=False, no_dupl=False)
    trafo3 = Trafo101Step(type=True, no_dupl=True)
    tokens = []
    tokens.append(model.Token(text="I", index_in_document=0,
                              pos_tag="PRP", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="am", index_in_document=1,
                              pos_tag="VBP", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="the", index_in_document=2,
                              pos_tag="DT", bio_tag="",
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

    tokens2 = copy.deepcopy(tokens)
    tokens2.pop()
    tokens2.append(model.Token(text="available", index_in_document=8,
                               pos_tag="JJ", bio_tag="",
                               sentence_index=0))
    tokens2.append(model.Token(text=".", index_in_document=9,
                               pos_tag=".", bio_tag="",
                               sentence_index=0))
    sentence2 = model.Sentence(tokens=tokens2)

    mention1 = model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3])
    mention2 = model.Mention(ner_tag="Further Specification", sentence_index=1, token_indices=[4, 5])
    doc = model.Document(
        text="I am the Head of the functional department.I am the Head of the functional department available.",
        name="1", sentences=[sentence1, sentence2],
        mentions=[mention1, mention2],
        entities=[],
        relations=[])
    doc_sol = copy.deepcopy(doc)
    doc_sol.sentences[0].tokens[6].text = "running"
    doc_sol.sentences[1].tokens[6].text = "running"

    doc_sol2 = copy.deepcopy(doc)
    doc_sol2.sentences[0].tokens[3].text = "proletarian"
    doc_sol2.sentences[1].tokens[3].text = "proletarian"

    doc_sol3 = copy.deepcopy(doc)
    doc_sol3.sentences[0].tokens[6].text = "running"
    doc_sol3.sentences[1].tokens[6].text = "functional"

    # Act
    doc_aug = trafo.do_augment(doc)
    doc_aug2 = trafo2.do_augment(doc)
    doc_aug3 = trafo3.do_augment(doc)

    # Assert
    assert doc_aug == doc_sol
    assert doc_aug2 == doc_sol2
    assert doc_aug3 == doc_sol3
