import json

from data import model
import typing
import data

docs: typing.List[model.Document] = data.loader.read_documents_from_json('../../complete.json')

def generate():
    with open(
            "compound_paraphrases_semeval2013_task4.json"
            ,
            "r",
    ) as fd:
        compounds = json.load(fd)

    for doc in docs:
        for sentence in doc.sentences:
            sentence_text = ""
            for token in sentence.tokens:
                sentence_text += token.text + " "
                for k in compounds.keys():
                    if k in sentence_text:
                        print(k)
                        print(token.text)









generate()