from data import model
import typing
import data
import nltk
from nltk.corpus import wordnet
import json
import numpy as np
docs: typing.List[model.Document] = data.loader.read_documents_from_json('../../one_doc.json')

def generate(prob=0.5):

    for doc in docs:
        for sentence in doc.sentences:
            for token in sentence.tokens:
                if (token.pos_tag == "JJ" or token.pos_tag == "JJS" or token.pos_tag == "JJR") and np.random.random() < prob :
                    syns = wordnet.synsets(token.text, "s")
                    syns = [syn.name().split(".")[0] for syn in syns]
                    for i in range(len(syns)):
                        if token.text != syns[i]:
                            token.text = syns[i]
                            break

#nltk.download()
generate()
json_data = [doc.to_json_serializable() for doc in docs]

with open("./test_101.json", "w") as f:
    json.dump(json_data, f)