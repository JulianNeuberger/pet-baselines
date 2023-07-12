import copy
from augment.trafo58 import Trafo58Step
from data import model
import mockito
from augment import trafo58

# Author for entire script: Benedikt
def test_do_augment():
    # Arrange
    # Testfall 1: ADj, Nomen und Verben auf Sprache Chinesisch
    trafo1 = Trafo58Step(True, True, True, "zh")
    tokens1 = [model.Token(text="I", index_in_document=0,
                           pos_tag="PRP", bio_tag="B-Actor",
                           sentence_index=0),
               model.Token(text="leave", index_in_document=1,
                           pos_tag="VBP", bio_tag="O",
                           sentence_index=0),
               model.Token(text="Human", index_in_document=2,
                           pos_tag="NN",
                           bio_tag="B-Activity Data",
                           sentence_index=0),
               model.Token(text="resources", index_in_document=3,
                           pos_tag="NNS", bio_tag="I-Activity Data",
                           sentence_index=0),
               model.Token(text="easy", index_in_document=4,
                           pos_tag="JJ", bio_tag="O",
                           sentence_index=0),
               model.Token(text=".", index_in_document=5,
                           pos_tag=".", bio_tag=".",
                           sentence_index=0)
               ]

    sentence1 = model.Sentence(tokens=tokens1)
    doc1 = model.Document(
        text="I leave Human resources",
        name="1", sentences=[sentence1],
        mentions=[],
        entities=[],
        relations=[])
    tokens1_sol = [model.Token(text="I", index_in_document=0,
                               pos_tag="PRP", bio_tag="B-Actor",
                               sentence_index=0),
                   model.Token(text="holidays", index_in_document=1,
                               pos_tag="NNS", bio_tag="O",
                               sentence_index=0),
                   model.Token(text="humanities", index_in_document=2,
                               pos_tag="NNS",
                               bio_tag="B-Activity Data",
                               sentence_index=0),
                   model.Token(text="resources", index_in_document=3,
                               pos_tag="NNS", bio_tag="I-Activity Data",
                               sentence_index=0),
                   model.Token(text="for", index_in_document=4,
                               pos_tag="IN", bio_tag="I-Activity Data",
                               sentence_index=0),
                   model.Token(text="resource", index_in_document=5,
                               pos_tag="NN", bio_tag="I-Activity Data",
                               sentence_index=0),
                   model.Token(text="resources", index_in_document=6,
                               pos_tag="NNS", bio_tag="I-Activity Data",
                               sentence_index=0),
                   model.Token(text="simple", index_in_document=7,
                               pos_tag="NN", bio_tag="O",
                               sentence_index=0),
                   model.Token(text=".", index_in_document=8,
                               pos_tag=".", bio_tag=".",
                               sentence_index=0)]
    sentence1_sol = model.Sentence(tokens=tokens1_sol)
    doc1_sol = model.Document(
        text="I leave Human resources",
        name="1", sentences=[sentence1_sol],
        mentions=[],
        entities=[],
        relations=[])
    doc_to_aug1 = copy.deepcopy(doc1)



    # Act
    doc_aug1 = trafo1.do_augment(doc_to_aug1)


    # Assert
    assert doc_aug1 == doc1_sol
