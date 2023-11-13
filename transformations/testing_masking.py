import copy
import typing
from transformers import pipeline

import data
from data import model

docs: typing.List[model.Document] = data.loader.read_documents_from_json('../complete.json')
def mask(strr):
    unmasker = pipeline(
        "fill-mask", model="xlm-roberta-base", top_k=1
    )
    masked_input = strr.replace(strr, "<mask>", 1)
    new_str = unmasker("I <mask> to read a <mask>")
    new_words = []
    for nw in new_str:
        new_words.append(nw[0]["token_str"])
    print(new_words)
#mask("make")


def do_augment(doc2: model.Document) -> model.Document:
    unmasker = pipeline(
        "fill-mask", model="xlm-roberta-base", top_k=1
    )
    doc = copy.deepcopy(doc2)
    for sentence in doc.sentences:
        sentence_string = ""
        changed_id = []
        for i, token in enumerate(sentence.tokens):
            if token.pos_tag in ["NN"]:
                sentence_string += " <mask>"
                changed_id.append(i)
            else:
                sentence_string += f" {token.text}"
        if len(changed_id) == 0:
            continue
        new_sent = unmasker(sentence_string)
        new_words = []
        for nw in new_sent:
            if len(changed_id) == 1:
                new_words.append(nw["token_str"])
            else:
                new_words.append(nw[0]["token_str"])
        for j, i in enumerate(changed_id):
            sentence.tokens[i].text = new_words[j]
    return doc

print(do_augment(docs[0]))