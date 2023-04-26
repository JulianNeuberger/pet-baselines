# 96 Subject Object Switch
from data import model
import typing
import data
import json

#Dataset
docs: typing.List[model.Document]= data.loader.read_documents_from_json('../trafo_test_doc.json')

def generate():
    for doc in docs:
        for sentence in doc.sentences:
            for token in sentence.tokens:
                pass
