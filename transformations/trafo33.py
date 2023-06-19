import copy
import random
import re
from collections import defaultdict

from data import model
from transformations import tokenmanager

MARKER_TO_CLASS = {
    "accordingly,": "Contingency.Cause.Result",
    "additionally,": "Expansion.Conjunction",
    "also,": "Expansion.Conjunction",
    "and,": "Expansion.Conjunction",
    "as a result": "Contingency.Cause.Result",
    "at the same time": "Temporal.Synchrony",
    "at the time": "Temporal.Synchrony",
    "because": "Contingency.Cause.Reason",
    "besides": "Expansion.Conjunction",
    "but,": "Comparison.Contrast",
    "consequently": "Contingency.Cause.Result",
    "earlier": "Temporal.Asynchronous.Succession",
    "finally": "Temporal.Asynchronous.Precedence",
    "for example": "Expansion.Instantiation",
    "for instance": "Expansion.Instantiation",
    "further": "Expansion.Conjunction",
    "furthermore": "Expansion.Conjunction",
    "hence": "Contingency.Cause.Result",
    "however,": "Comparison.Contrast",
    "in addition,": "Expansion.Conjunction",
    "in fact,": "Expansion.Conjunction",
    "in particular": "Expansion.Restatement.Specification",
    "in turn,": "Expansion.Conjunction",
    "inasmuch as": "Contingency.Cause.Reason",
    "later": "Temporal.Asynchronous.Precedence",
    "likewise": "Expansion.Conjunction",
    "meanwhile": "Expansion.Conjunction",
    "moreover": "Expansion.Conjunction",
    "on the contrary": "Comparison.Contrast",
    "previously": "Temporal.Asynchronous.Succession",
    "similarly": "Expansion.Conjunction",
    "since": "Contingency.Cause.Reason",
    "so,": "Contingency.Cause.Result",
    "specifically": "Expansion.Restatement.Specification",
    "subsequently": "Temporal.Asynchronous.Precedence",
    "then,": "Temporal.Asynchronous.Precedence",
    "therefore": "Contingency.Cause.Result",
    "thus": "Contingency.Cause.Result",
}

CLASS_TO_MARKERS = defaultdict(list)
for (k, v) in MARKER_TO_CLASS.items():
    CLASS_TO_MARKERS[v].append(k.rstrip(","))
CLASS_TO_MARKERS = dict(CLASS_TO_MARKERS)


def generate(doc: model.Document):
    together = separate()

    old_list = together[0]
    new_list = together[1]
    for sentence in doc.sentences:
        i = 0
        while i < len(sentence.tokens):
            print("----------")
            print(sentence.tokens[i].text)
            token = sentence.tokens[i]
            for k in range(len(old_list)):
                marker_list = old_list[k].replace(",", " ").split()
                if marker_list[0] == token.text:
                    is_equal = True
                    if len(marker_list) > 1:
                        #y = copy.deepcopy(i)
                        for j in range(1, len(marker_list)):
                            if i + j < len(sentence.tokens):

                                if sentence.tokens[i + j].text != marker_list[j]:
                                    print(sentence.tokens[i + j].text)
                                    print(marker_list[j])

                                    is_equal = False
                                    continue
                                    #i += 1
                    print(is_equal)
                    if is_equal:
                        print(new_list[k])
                        new = new_list[k][1].split()
                        print(new)
                        if new == marker_list:
                            pass
                        for j in range(len(new)):
                            bio_t = token.bio_tag

                            if j < len(marker_list):
                                print("test")
                                print(i)
                                print(j)

                                if i + j < len(sentence.tokens):

                                    sentence.tokens[i + j].text = new[j]
                                    sentence.tokens[i + j].pos_tag = tokenmanager.get_pos_tag([new[j]])
                                    sentence.tokens[i + j].bio_tag = bio_t
                                if j != 0:
                                    pass
                                    #i += 1
                            if j >= len(marker_list):
                                tok = model.Token(text=new[j], index_in_document=token.index_in_document + j,
                                                  pos_tag=tokenmanager.get_pos_tag([new[j]]),
                                                  bio_tag=bio_t,
                                                  sentence_index=token.sentence_index)
                                print("tok text")
                                print(tok.text)
                                print(i + j)
                                tokenmanager.create_token(doc, tok, i + j)
                                i += 1
                            if len(marker_list) > len(new):
                                num_of_changes = len(marker_list) - len(new)
                                for i in range(num_of_changes):
                                    tokenmanager.delete_token(doc, token.index_in_document + len(new))

            i += 1
    return doc




def separate():
    before = []
    after = []
    for m in MARKER_TO_CLASS:
        before.append(m)
        after.append(CLASS_TO_MARKERS[MARKER_TO_CLASS[m]])
    return before, after


tokens = [model.Token(text="for", index_in_document=0,
                          pos_tag="JJ", bio_tag="O",
                          sentence_index=0),
              model.Token(text="example", index_in_document=1,
                          pos_tag="VBP", bio_tag="O",
                          sentence_index=0),
              model.Token(text="contrary", index_in_document=2,
                          pos_tag="NN", bio_tag="O",
                          sentence_index=0),
              model.Token(text="head", index_in_document=3,
                          pos_tag=".", bio_tag="O",
                          sentence_index=0),
          model.Token(text=".", index_in_document=4,
                      pos_tag=".", bio_tag="O",
                      sentence_index=0)
          ]
sentence1 = model.Sentence(tokens=tokens)

doc2 = model.Document(
    text="good leave head .",
    name="1", sentences=[sentence1],
    mentions=[],
    entities=[],
    relations=[])

print(generate(doc2))
#separate()