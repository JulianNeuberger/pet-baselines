import copy

import pytest

from augment.trafo3 import Trafo3Step
from data import model


def give_back_fixture(fix):
    return fix

@pytest.fixture
def two_sent_three_adjective_dupl_two_mentions():
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
    tokens.append(model.Token(text="Head", index_in_document=3,
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

    tokens2 = tokens.copy()
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
    doc = model.Document(text="I am the Head of the functional department.I am the Head of the functional department available.",
                          name="1", sentences=[sentence1, sentence2],
                          mentions=[mention1, mention2],
                          entities=[],
                          relations=[])
    return doc


@pytest.fixture()
def two_sent_no_adjectives_two_mentions():
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
    tokens.append(model.Token(text="Head", index_in_document=3,
                              pos_tag="NN", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="of", index_in_document=4,
                              pos_tag="IN", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="the", index_in_document=5,
                              pos_tag="DT", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="department", index_in_document=6,
                              pos_tag="NN", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text=".", index_in_document=7,
                              pos_tag=".", bio_tag="",
                              sentence_index=0))
    sentence = model.Sentence(tokens=tokens)
    mention1 = model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3])
    mention2 = model.Mention(ner_tag="Further Specification", sentence_index=1, token_indices=[4, 5])
    doc = model.Document(text="I am the Head of the department.", name="1", sentences=[sentence, sentence],
                         mentions=[mention1, mention2],
                         entities=[],
                         relations=[])
    return doc

def test_do_augment():
    # Arrange
    trafo1 = Trafo3Step(False, 1)
    trafo2 = Trafo3Step(True, 1)
    trafo3 = Trafo3Step(False, 2)

    # no adjectives
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
    tokens.append(model.Token(text="Head", index_in_document=3,
                              pos_tag="NN", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="of", index_in_document=4,
                              pos_tag="IN", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="the", index_in_document=5,
                              pos_tag="DT", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="department", index_in_document=6,
                              pos_tag="NN", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text=".", index_in_document=7,
                              pos_tag=".", bio_tag="",
                              sentence_index=0))
    sentence = model.Sentence(tokens=tokens)
    mention1 = model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3])
    mention2 = model.Mention(ner_tag="Further Specification", sentence_index=1, token_indices=[4, 5])
    doc1 = model.Document(text="I am the Head of the department.", name="1", sentences=[sentence, sentence],
                         mentions=[mention1, mention2],
                         entities=[],
                         relations=[])

    # NO DUPLICATES = False: duplicates allowed
    tokens2 = []
    tokens2.append(model.Token(text="I", index_in_document=0,
                              pos_tag="PRP", bio_tag="",
                              sentence_index=0))
    tokens2.append(model.Token(text="am", index_in_document=1,
                              pos_tag="VBP", bio_tag="",
                              sentence_index=0))
    tokens2.append(model.Token(text="the", index_in_document=2,
                              pos_tag="DT", bio_tag="",
                              sentence_index=0))
    tokens2.append(model.Token(text="Head", index_in_document=3,
                              pos_tag="NN", bio_tag="",
                              sentence_index=0))
    tokens2.append(model.Token(text="of", index_in_document=4,
                              pos_tag="IN", bio_tag="",
                              sentence_index=0))
    tokens2.append(model.Token(text="the", index_in_document=5,
                              pos_tag="DT", bio_tag="",
                              sentence_index=0))
    tokens2.append(model.Token(text="functional", index_in_document=6,
                              pos_tag="JJ", bio_tag="",
                              sentence_index=0))
    tokens2.append(model.Token(text="department", index_in_document=7,
                              pos_tag="NN", bio_tag="",
                              sentence_index=0))
    tokens2.append(model.Token(text=".", index_in_document=8,
                              pos_tag=".", bio_tag="",
                              sentence_index=0))
    sentence1 = model.Sentence(tokens=tokens2)

    tokens3 = copy.deepcopy(tokens2)
    tokens3.pop()
    tokens3.append(model.Token(text="available", index_in_document=8,
                               pos_tag="JJ", bio_tag="",
                               sentence_index=0))
    tokens3.append(model.Token(text=".", index_in_document=9,
                               pos_tag=".", bio_tag="",
                               sentence_index=0))
    sentence2 = model.Sentence(tokens=tokens3)

    mention1 = model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3])
    mention2 = model.Mention(ner_tag="Further Specification", sentence_index=1, token_indices=[4, 5])
    doc2 = model.Document(
        text="I am the Head of the functional department.I am the Head of the functional department available.",
        name="1", sentences=[sentence1, sentence2],
        mentions=[mention1, mention2],
        entities=[],
        relations=[])

    doc_sol2 = copy.deepcopy(doc2)
    doc_sol2.sentences[0].tokens[6].text = "nonfunctional"
    doc_sol2.sentences[1].tokens[6].text = "nonfunctional"

    # NO DUPLICATES = True: duplicates not allowed
    doc_sol3 = copy.deepcopy(doc2)
    doc_sol3.sentences[0].tokens[6].text = "nonfunctional"
    doc_sol3.sentences[1].tokens[8].text = "unavailable"

    # max number of replaced adjectives = 2 instead of 1
    doc_sol4 = copy.deepcopy(doc_sol3)
    doc_sol4.sentences[1].tokens[6].text = "nonfunctional"

    # Act
    doc_to_aug1 = copy.deepcopy(doc1)
    doc_aug1 = trafo1.do_augment(doc_to_aug1)

    doc_to_aug2 = copy.deepcopy(doc2)
    doc_aug2 = trafo1.do_augment(doc_to_aug2)

    doc_to_aug3 = copy.deepcopy(doc2)
    doc_aug3 = trafo2.do_augment(doc_to_aug3)

    doc_to_aug4 = copy.deepcopy(doc2)
    doc_aug4 = trafo3.do_augment(doc_to_aug4)

    # Assert
    assert doc_aug1 == doc1
    assert doc_aug2 == doc_sol2
    assert doc_aug3 == doc_sol3
    assert doc_aug4 == doc_sol4