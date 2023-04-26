#Transformation 3 Add Phrases

from data import model
import typing
import data
import json

#zugehörige Jsons
#trafo_test_doc.json als source

#Dataset
docs: typing.List[model.Document]= data.loader.read_documents_from_json('../trafo_test_doc.json')

def generate(prob=0.166, sp_p=True, unc_p=True, fill_p=True, max_outputs = 1):
    # Speaker opinion/mental state phrases
    # Taken from Kovatchev et al. (2021)
    speaker_phrases = [
        "I think",
        "I believe",
        "I mean",
        "I guess",
        "that is",
        "I assume",
        "I feel",
        "In my opinion",
        "I would say",
    ]

    # Words and phrases indicating uncertainty
    # Taken from Kovatchev et al. (2021)
    uncertain_phrases = [
        "maybe",
        "perhaps",
        "probably",
        "possibly",
        "most likely",
    ]

    # Filler words that should preserve the meaning of the phrase
    # Taken from Laserna et al. (2014)
    fill_phrases = [
        "uhm",
        "umm",
        "ahh",
        "err",
        "actually",
        "obviously",
        "naturally",
        "like",
        "you know",
    ]

    # Initialize the list of all augmentation phrases
    all_fill = []

    # Add speaker phrases, if enabled
    if sp_p:
        all_fill += speaker_phrases

    # Add uncertain phrases, if enabled
    if unc_p:
        all_fill += uncertain_phrases

    # Add filler phrases, if enabled
    if fill_p:
        all_fill += fill_phrases

    # Calculate probability of insertion, default is 16.6 (one in 6 words)
    prob_of_insertion = int(prob * 100)

    for doc in docs:
        for sentence in doc.sentences:
            #hier die logik
            pass