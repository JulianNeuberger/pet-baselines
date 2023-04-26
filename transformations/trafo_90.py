#Shuffle within Segments 90

from data import model
import typing
import data
import itertools
import json
import numpy as np


#zugehörige Jsons
#trafo_test_doc.json als source

#Dataset
docs: typing.List[model.Document]= data.loader.read_documents_from_json('../trafo_test_doc.json')


def generate():
    #np.random.seed(self.seed)

    for doc in docs:
        for sentence in doc.sentences:
            token_seq = []
            tag_seq = []
            # generate token and tag list
            for token in sentence.tokens:
                print(token.text, end= ' ')
                token_seq.append(token.text)
                tag_seq.append(token.bio_tag)
            # compare whether both lists have the same length - if not error
            assert len(token_seq) == len(
                tag_seq
            ), "Lengths of token sequence and BIO-tag sequence should be the same"
            # we need the original indices of each tag - (indice, tag)
            tags = [(i, t) for i, t in enumerate(tag_seq)]
            # split tags into groups - [(indice, tag), (),...]
            groups = [
                list(g)
                for k, g in itertools.groupby(tags, lambda s: s[1].split("-")[-1])
            ]
            #shuffle tokens in groups
            pos = 0 # position des in dem ursprünglichen Satz zu ersetzenden tokens
            print("")
            for group in groups:
                indices = [i[0] for i in group]
                if np.random.binomial(1, 0.5):
                    np.random.shuffle(indices)
                for i in range(len(group)):
                    sentence.tokens[pos].text = token_seq[indices[i]]
                    print(sentence.tokens[pos].text, end= ' ')
                    pos += 1
            print("")

generate()
json_data = [doc.to_json_serializable() for doc in docs]
with open("./trafo_test.json", "w") as f:
    json.dump(json_data, f)


