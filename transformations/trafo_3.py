#Transformation 3 Antonyms

from data import model
import typing
import data
import nltk
from nltk.corpus import wordnet
import json

#zugehörige Jsons
#trafo_test_doc.json als source, trafo_test.json als Ziel

#Dataset
docs: typing.List[model.Document]= data.loader.read_documents_from_json('../trafo_test_doc.json')

def generate():
    for doc in docs:
        for sentence in doc.sentences:
            for token in sentence.tokens:
                if token.pos_tag == "JJ" or token.pos_tag == "JJS" or token.pos_tag == "JJR": #bei denen ADJ
                    #Get Synsets
                    synsets = wordnet.synsets(token.text, "a")

                    # Get Antonyms
                    if synsets:
                        first_synset = synsets[0]
                        lemmas = first_synset.lemmas()
                        first_lemma = lemmas[0]
                        antonyms = first_lemma.antonyms()

                    # Get first Antonym
                    if antonyms:
                        antonyms.sort(key=lambda x: str(x).split(".")[2])
                        first_antonym = antonyms[0].name()
                    print("--------")
                    print(token.text)
                    print(first_antonym)
                    #Replace adjective with antonym
                    token.text = first_antonym


generate()
json_data = [doc.to_json_serializable() for doc in docs]
with open("./trafo_test.json", "w") as f:
    json.dump(json_data, f)