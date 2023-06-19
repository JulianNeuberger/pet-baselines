from transformations.tokenmanager import get_pos_tag, get_bio_tag_based_on_left_token, insert_token_in_mentions, \
    create_token, get_index_in_sentence, get_mentions, insert_token_in_tokens, summe
from data import model
import mockito



def test_get_pos_tag():  # passed
    # Arrange
    p1 = []
    p2 = ["my"]
    p3 = ["my", "name", "is"]

    # Act
    t1 = get_pos_tag(p1)
    t2 = get_pos_tag(p2)
    t3 = get_pos_tag(p3)

    # Assert
    assert t1 == []
    assert t2 == ['PRP$']
    assert t3 == ['PRP$', 'NN', 'VBZ']


def test_get_bio_tag_based_on_left_token():  # passed
    # Arrange
    p1 = "O"
    p2 = "B-Actor"
    p3 = "I-Actor"

    # Act
    t1 = get_bio_tag_based_on_left_token(p1)
    t2 = get_bio_tag_based_on_left_token(p2)
    t3 = get_bio_tag_based_on_left_token(p3)

    # Assert
    assert t1 == p1
    assert t2 == p3
    assert t3 == p3


def test_get_index_in_sentence():  # passed
    # Arrange

    # sentence does not exist
    sent1 = model.Sentence(tokens=[])
    text1 = ["I", "am"]

    # text does not exist
    toks2 = []
    toks2.append(model.Token(text="I", index_in_document=0, pos_tag="", bio_tag="", sentence_index=0))

    sent2 = model.Sentence(tokens=toks2)
    text2 = []

    # both exist, text is not in sentence
    toks3 = []
    toks3.append(model.Token(text="I", index_in_document=0, pos_tag="", bio_tag="", sentence_index=0))
    toks3.append(model.Token(text="am", index_in_document=0,
                             pos_tag="", bio_tag="",
                             sentence_index=0))
    toks3.append(model.Token(text="the", index_in_document=0,
                             pos_tag="", bio_tag="",
                             sentence_index=0))
    toks3.append(model.Token(text="best", index_in_document=0,
                             pos_tag="", bio_tag="",
                             sentence_index=0))
    sent3 = model.Sentence(tokens=toks3)
    text3 = ["am", "the", "worst"]

    # both exist, text is in sentence
    sent4 = model.Sentence(tokens=toks3)
    text4 = ["am", "the"]

    # Act
    i1 = get_index_in_sentence(sent1, text1)
    i2 = get_index_in_sentence(sent2, text2)
    i3 = get_index_in_sentence(sent3, text3)
    i4 = get_index_in_sentence(sent4, text4)

    # Assert
    assert i1 is None
    assert i2 is None
    assert i3 is None
    assert i4 == 1


def test_get_mentions():  # passed
    # Arrange
    tokens = []
    tokens.append(model.Token(text="I", index_in_document=0,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="am", index_in_document=1,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="the", index_in_document=2,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="chief", index_in_document=3,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="of", index_in_document=4,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="the", index_in_document=5,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="department", index_in_document=6,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text=".", index_in_document=7,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    sentence = model.Sentence(tokens=tokens)
    mention = model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3])
    doc = model.Document(text="I am the chief of the department.", name="1", sentences=[sentence], mentions=[mention],
                         entities=[],
                         relations=[])

    # token is not in mentions
    ind_in_sent1 = 5
    sent_ind1 = 0

    # token is in mentions
    ind_in_sent2 = 2
    sent_ind2 = 0

    # Act
    ment1 = get_mentions(doc, ind_in_sent1, sent_ind1)
    ment2 = get_mentions(doc, ind_in_sent2, sent_ind2)

    # Assert
    assert ment1 == []
    assert ment2 == [0]


def test_insert_token_in_mentions():  # passed
    # Arrange
    tokens = []
    tokens.append(model.Token(text="I", index_in_document=0,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="am", index_in_document=1,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="the", index_in_document=2,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="chief", index_in_document=3,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="of", index_in_document=4,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="the", index_in_document=5,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="department", index_in_document=6,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text=".", index_in_document=7,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    sentence = model.Sentence(tokens=tokens)
    mention = model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3])
    doc = model.Document(text="I am the chief of the department.", name="1", sentences=[sentence], mentions=[mention],
                         entities=[],
                         relations=[])
    # mention_id does not exist (index_in_sentence is always ok due to the prior method)
    ment_id1 = 2
    index_in_sent1 = 5

    # mention_id exists
    ment_id2 = 0
    index_in_sent2 = 5
    mention = model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3, 5])
    doc2 = model.Document(text="I am the chief of the department.", name="1", sentences=[sentence], mentions=[mention],
                          entities=[],
                          relations=[])

    # no duplicate index_in_sentence_id
    ment_id3 = 0
    index_in_sent3 = 5

    # Act
    doc1 = doc.copy()
    insert_token_in_mentions(doc1, index_in_sent1, ment_id1)

    doc3 = doc.copy()
    insert_token_in_mentions(doc3, index_in_sent2, ment_id2)

    doc4 = doc2.copy()
    insert_token_in_mentions(doc4, index_in_sent3, ment_id3)

    # Assert
    assert doc1 == doc
    assert doc3 == doc2
    assert doc4 == doc2


def test_insert_token_in_tokens():  # passed
    # Arrange
    tokens = []
    tokens.append(model.Token(text="I", index_in_document=0,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="am", index_in_document=1,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="the", index_in_document=2,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="chief", index_in_document=3,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="of", index_in_document=4,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="the", index_in_document=5,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="department", index_in_document=6,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text=".", index_in_document=7,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    sentence = model.Sentence(tokens=tokens)

    tokens1 = []
    tokens1.append(model.Token(text="I", index_in_document=8,
                               pos_tag="", bio_tag="",
                               sentence_index=1))
    tokens1.append(model.Token(text="can", index_in_document=9,
                               pos_tag="", bio_tag="",
                               sentence_index=1))
    tokens1.append(model.Token(text="do", index_in_document=10,
                               pos_tag="", bio_tag="",
                               sentence_index=1))
    tokens1.append(model.Token(text="everything", index_in_document=11,
                               pos_tag="", bio_tag="",
                               sentence_index=1))
    tokens1.append(model.Token(text=".", index_in_document=12,
                               pos_tag="", bio_tag="",
                               sentence_index=1))
    sentence1 = model.Sentence(tokens=tokens1)

    tokens2 = []
    tokens2.append(model.Token(text="The", index_in_document=13,
                               pos_tag="", bio_tag="",
                               sentence_index=2))
    tokens2.append(model.Token(text="job", index_in_document=14,
                               pos_tag="", bio_tag="",
                               sentence_index=2))
    tokens2.append(model.Token(text="is", index_in_document=15,
                               pos_tag="", bio_tag="",
                               sentence_index=2))
    tokens2.append(model.Token(text="nice", index_in_document=16,
                               pos_tag="", bio_tag="",
                               sentence_index=2))
    tokens2.append(model.Token(text=".", index_in_document=17,
                               pos_tag="", bio_tag="",
                               sentence_index=2))
    sentence2 = model.Sentence(tokens=tokens2)

    mention = model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3])
    mention1 = model.Mention(ner_tag="Actor", sentence_index=1, token_indices=[0, 1])
    mention2 = model.Mention(ner_tag="Actor", sentence_index=2, token_indices=[4])
    doc = model.Document(text="I am the chief of the department. I can do everything. The job is nice.", name="1",
                         sentences=[sentence, sentence1, sentence2], mentions=[mention, mention1, mention2],
                         entities=[],
                         relations=[])

    # index in sentence is not correct
    tok1 = model.Token(text="The", index_in_document=0,
                       pos_tag="", bio_tag="",
                       sentence_index=0)
    ind_in_sent1 = -1

    # token.sentence_index is not correct
    tok2 = model.Token(text="The", index_in_document=0,
                       pos_tag="", bio_tag="",
                       sentence_index=3)
    ind_in_sent2 = 1

    # both are correct, token hast to be inserted
    tok3 = model.Token(text="Hallo", index_in_document=9,
                       pos_tag="", bio_tag="",
                       sentence_index=1)
    ind_in_sent3 = 1
    tokens3 = []
    tokens3.append(model.Token(text="I", index_in_document=8,
                               pos_tag="", bio_tag="",
                               sentence_index=1))
    tokens3.append(tok3)
    tokens3.append(model.Token(text="can", index_in_document=10,
                               pos_tag="", bio_tag="",
                               sentence_index=1))
    tokens3.append(model.Token(text="do", index_in_document=11,
                               pos_tag="", bio_tag="",
                               sentence_index=1))
    tokens3.append(model.Token(text="everything", index_in_document=12,
                               pos_tag="", bio_tag="",
                               sentence_index=1))
    tokens3.append(model.Token(text=".", index_in_document=13,
                               pos_tag="", bio_tag="",
                               sentence_index=1))
    sentence3 = model.Sentence(tokens=tokens3)

    tokens4 = []
    tokens4.append(model.Token(text="The", index_in_document=14,
                               pos_tag="", bio_tag="",
                               sentence_index=2))
    tokens4.append(model.Token(text="job", index_in_document=15,
                               pos_tag="", bio_tag="",
                               sentence_index=2))
    tokens4.append(model.Token(text="is", index_in_document=16,
                               pos_tag="", bio_tag="",
                               sentence_index=2))
    tokens4.append(model.Token(text="nice", index_in_document=17,
                               pos_tag="", bio_tag="",
                               sentence_index=2))
    tokens4.append(model.Token(text=".", index_in_document=18,
                               pos_tag="", bio_tag="",
                               sentence_index=2))
    sentence4 = model.Sentence(tokens=tokens4)

    mention1 = model.Mention(ner_tag="Actor", sentence_index=1, token_indices=[0, 2])
    doc3 = model.Document(text="I am the chief of the department. I can do everything. The job is nice.", name="1",
                          sentences=[sentence, sentence3, sentence4], mentions=[mention, mention1, mention2],
                          entities=[], relations=[])

    # Act
    doc1 = doc.copy()
    insert_token_in_tokens(doc1, tok1, ind_in_sent1)

    doc2 = doc.copy()
    insert_token_in_tokens(doc2, tok2, ind_in_sent2)

    doc4 = doc.copy()
    insert_token_in_tokens(doc4, tok3, ind_in_sent3)

    # Assert
    assert doc1 == doc
    assert doc2 == doc
    assert doc3 == doc4


def test_create_token():  # passed

    # Arrange
    tokens = []
    tokens.append(model.Token(text="I", index_in_document=0,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="am", index_in_document=1,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="the", index_in_document=2,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="chief", index_in_document=3,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="of", index_in_document=4,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="the", index_in_document=5,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text="department", index_in_document=6,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    tokens.append(model.Token(text=".", index_in_document=7,
                              pos_tag="", bio_tag="",
                              sentence_index=0))
    sentence = model.Sentence(tokens=tokens)

    tokens1 = []
    tokens1.append(model.Token(text="I", index_in_document=8,
                               pos_tag="", bio_tag="",
                               sentence_index=1))
    tokens1.append(model.Token(text="can", index_in_document=9,
                               pos_tag="", bio_tag="",
                               sentence_index=1))
    tokens1.append(model.Token(text="do", index_in_document=10,
                               pos_tag="", bio_tag="",
                               sentence_index=1))
    tokens1.append(model.Token(text="everything", index_in_document=11,
                               pos_tag="", bio_tag="",
                               sentence_index=1))
    tokens1.append(model.Token(text=".", index_in_document=12,
                               pos_tag="", bio_tag="",
                               sentence_index=1))
    sentence1 = model.Sentence(tokens=tokens1)

    tokens2 = []
    tokens2.append(model.Token(text="The", index_in_document=13,
                               pos_tag="", bio_tag="",
                               sentence_index=2))
    tokens2.append(model.Token(text="job", index_in_document=14,
                               pos_tag="", bio_tag="",
                               sentence_index=2))
    tokens2.append(model.Token(text="is", index_in_document=15,
                               pos_tag="", bio_tag="",
                               sentence_index=2))
    tokens2.append(model.Token(text="nice", index_in_document=16,
                               pos_tag="", bio_tag="",
                               sentence_index=2))
    tokens2.append(model.Token(text=".", index_in_document=17,
                               pos_tag="", bio_tag="",
                               sentence_index=2))
    sentence2 = model.Sentence(tokens=tokens2)

    mention = model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3])
    mention1 = model.Mention(ner_tag="Actor", sentence_index=1, token_indices=[0, 1])
    mention2 = model.Mention(ner_tag="Actor", sentence_index=2, token_indices=[4])
    doc = model.Document(text="I am the chief of the department. I can do everything. The job is nice.", name="1",
                         sentences=[sentence, sentence1, sentence2], mentions=[mention, mention1, mention2],
                         entities=[],
                         relations=[])

    # index in sentence is not valid
    ind_in_sent1 = 8
    tok = model.Token(text="Hallo", index_in_document=9,
                      pos_tag="", bio_tag="",
                      sentence_index=1)
    ment_ind1 = None

    # mention id is not valid
    ind_in_sent2 = 2
    ment_ind2 = 5

    # both ids are valid, don't change mention
    ind_in_sent3 = 1

    tokens3 = []
    tokens3.append(model.Token(text="I", index_in_document=8,
                               pos_tag="", bio_tag="",
                               sentence_index=1))
    tokens3.append(tok)
    tokens3.append(model.Token(text="can", index_in_document=10,
                               pos_tag="", bio_tag="",
                               sentence_index=1))
    tokens3.append(model.Token(text="do", index_in_document=11,
                               pos_tag="", bio_tag="",
                               sentence_index=1))
    tokens3.append(model.Token(text="everything", index_in_document=12,
                               pos_tag="", bio_tag="",
                               sentence_index=1))
    tokens3.append(model.Token(text=".", index_in_document=13,
                               pos_tag="", bio_tag="",
                               sentence_index=1))
    sentence3 = model.Sentence(tokens=tokens3)

    tokens4 = []
    tokens4.append(model.Token(text="The", index_in_document=14,
                               pos_tag="", bio_tag="",
                               sentence_index=2))
    tokens4.append(model.Token(text="job", index_in_document=15,
                               pos_tag="", bio_tag="",
                               sentence_index=2))
    tokens4.append(model.Token(text="is", index_in_document=16,
                               pos_tag="", bio_tag="",
                               sentence_index=2))
    tokens4.append(model.Token(text="nice", index_in_document=17,
                               pos_tag="", bio_tag="",
                               sentence_index=2))
    tokens4.append(model.Token(text=".", index_in_document=18,
                               pos_tag="", bio_tag="",
                               sentence_index=2))
    sentence4 = model.Sentence(tokens=tokens4)

    mention = model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3])
    mention2 = model.Mention(ner_tag="Actor", sentence_index=2, token_indices=[4])
    mention1 = model.Mention(ner_tag="Actor", sentence_index=1, token_indices=[0, 2])
    doc4 = model.Document(text="I am the chief of the department. I can do everything. The job is nice.", name="1",
                          sentences=[sentence, sentence3, sentence4], mentions=[mention, mention1, mention2],
                          entities=[], relations=[])

    # both are valid, don't change mention
    ind_in_sent4 = 1
    ment_ind4 = 1
    mention3 = model.Mention(ner_tag="Actor", sentence_index=1, token_indices=[0, 1, 2])
    doc6 = model.Document(text="I am the chief of the department. I can do everything. The job is nice.", name="1",
                          sentences=[sentence, sentence3, sentence4], mentions=[mention, mention3, mention2],
                          entities=[], relations=[])

    # Act
    doc1 = doc.copy()
    create_token(doc1, tok, ind_in_sent1, ment_ind1)

    doc2 = doc.copy()
    create_token(doc2, tok, ind_in_sent2, ment_ind2)

    doc3 = doc.copy()
    create_token(doc3, tok, ind_in_sent3, None)

    doc5 = doc.copy()
    create_token(doc5, tok, ind_in_sent4, ment_ind4)

    # Assert
    assert doc1 == doc
    assert doc2 == doc
    assert doc3 == doc4
    assert doc5 == doc6

def test_summe():
    # Act
    mockito.when()