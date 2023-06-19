import copy

from data import model
import typing
import data
import nltk
from nltk.corpus import wordnet
import json
import numpy as np
#docs: typing.List[model.Document] = data.loader.read_documents_from_json('../../one_doc.json')

def generate(doc, prob=0.5):

    #for doc in docs:
        for sentence in doc.sentences:
            for i in range(len(sentence.tokens)):
                token = sentence.tokens[i]
                if (token.pos_tag == "JJ" or token.pos_tag == "JJS" or token.pos_tag == "JJR") and np.random.random() < prob :
                    syns = wordnet.synsets(token.text, "s")
                    syns = [syn.name().split(".")[0] for syn in syns]
                    for i in range(len(syns)):
                        if token.text != syns[i]:
                            token.text = syns[i]
                            break

#nltk.download()


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

tokens2 = copy.deepcopy(tokens)
tokens2.pop()
tokens2.append(model.Token(text="functional", index_in_document=8,
                               pos_tag="JJ", bio_tag="",
                               sentence_index=0))
tokens2.append(model.Token(text=".", index_in_document=9,
                          pos_tag=".", bio_tag="",
                          sentence_index=0))
tokens2[6].text = "available"
sentence2 = model.Sentence(tokens=tokens2)

mention1 = model.Mention(ner_tag="Actor", sentence_index=0, token_indices=[2, 3])
mention2 = model.Mention(ner_tag="Further Specification", sentence_index=1, token_indices=[4, 5])
doc = model.Document(text="I am the Head of the functional department.I am the Head of the functional department available.",
                      name="1", sentences=[sentence1, sentence2],
                      mentions=[mention1, mention2],
                      entities=[],
                      relations=[])

generate(doc, 1)
print(doc)