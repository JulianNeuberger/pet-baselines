import copy

from augment import Filter10Step
from data import model


# Author for entire script: Benedikt
def test_do_augment():
    filter1 = Filter10Step(length=4)
    filter2 = Filter10Step(length=5)
    tokens = [model.Token(text="I", index_in_document=0, pos_tag="PRP", bio_tag="B-Activity", sentence_index=0),
              model.Token(text="am", index_in_document=1, pos_tag="VBP", bio_tag="B-Activity", sentence_index=0),
              model.Token(text="the", index_in_document=2, pos_tag="DT", bio_tag="B-Activity", sentence_index=0),
              model.Token(text="Head", index_in_document=3, pos_tag="NN", bio_tag="B-Activity", sentence_index=0),
              model.Token(text="of", index_in_document=4, pos_tag="IN", bio_tag="B-Activity", sentence_index=0),
              model.Token(text="the", index_in_document=5, pos_tag="DT", bio_tag="", sentence_index=0),
              model.Token(text="department", index_in_document=6, pos_tag="NN", bio_tag="", sentence_index=0),
              model.Token(text=".", index_in_document=7, pos_tag=".", bio_tag="", sentence_index=0)]

    sentence = model.Sentence(tokens=tokens)
    doc = model.Document(text="I am the Head of the department.", name="1", sentences=[sentence],
                          mentions=[], entities=[], relations=[])

    doc_sol = model.Document(text="I am the Head of the department.", name="1",sentences=[],
                             mentions=[], entities=[], relations=[])
    doc_sol2 = copy.deepcopy(doc)


    doc1 = filter1.do_augment(doc)
    doc2 = filter2.do_augment(doc)
    assert doc_sol == doc1
    assert doc_sol2 == doc2