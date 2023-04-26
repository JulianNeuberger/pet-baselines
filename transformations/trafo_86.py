#Transformation 86 Hypernym Hyponym

from data import model
import typing
import data
import random
import numpy as np
#from checklist.editor import Editor
import json

#zugehörige Jsons
#trafo_test_doc.json als source

#Dataset
docs: typing.List[model.Document]= data.loader.read_documents_from_json('../trafo_test_doc.json')

#parameter: max_output für Anzahl ausgegebener Sätze (bei uns eig nur 1 sinnvoll)
#           max_noun für Anzahl an zu verändernden Nomen (im original nur 1)
#           kind_of_replacement: 0 für hypernym, 1 für hypernym, 2 für zufällig

def generate(max_output=1, max_noun=1, kind_of_replacement=2):
    for doc in docs:
        for sentence in doc.sentences:

            # list with all tokens with pos_tag noun
            token_list = []
            for token in sentence.tokens:
                if token.pos_tag == "NN" or token.pos_tag == "NNS" or token.pos_tag == "NNP" or token.pos_tag == "NNPS":
                    token_list.append(token)

            # shuffle für zufällige Nomenauswahl
            random.shuffle(token_list)

            #je nomen suche hypernym oder hyponym
            for token in token_list:
                for i in range(1, max_noun):
                    if kind_of_replacement == 2:
                        num = np.random.random()
                        if num <= 0.5:
                            kind_of_replacement = 0
                        else:
                            kind_of_replacement = 1
                    if kind_of_replacement == 1:
                        #suche hypernym
                        #hyp_list = self.editor.hypernyms(sentence, token.text)
                        #if len(hyp_list) >= 1:
                         #   token.text = hyp_list[0]
                        pass
                    elif kind_of_replacement == 0:
                        #suche hyponym#
                        # hyp_list = self.editor.hyponyms(sentence, token.text)
                        # if len(hyp_list) >= 1:
                        #   token.text = hyp_list[0]
                        pass

                #Änderung des token.text schreiben
                for t in sentence.tokens:
                    if token.index_in_document == t.index_in_document:
                        t.text = token.text


generate()

#json_data = [doc.to_json_serializable() for doc in docs]
#with open("./trafo_test.json", "w") as f:
#    json.dump(json_data, f)