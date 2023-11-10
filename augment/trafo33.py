import typing
from random import randint, random
from augment import base, params
from data import model
from transformations import tokenmanager
from collections import defaultdict


# Author: Benedikt
# not used in final work
class Trafo33Step(base.AugmentationStep):
    def __init__(self, p: float = 1):
        self.p = p

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        return [params.FloatParam(name="p", min_value=0.0, max_value=1.0)]

    def do_augment(self, doc: model.Document):
        together = Trafo33Step.separate(self)
        # old list, to search the matching text in the Document, new list is the text that replaces the old text
        old_list = together[0]
        new_list = together[1]
        for sentence in doc.sentences:
            i = 0
            while i < len(sentence.tokens):
                token = sentence.tokens[i]
                #  search in old list for matching text in the document; when found, is_equal is set to true
                for k in range(len(old_list)):
                    marker_list = old_list[k].replace(",", " ").split()
                    if marker_list[0] == token.text:
                        is_equal = True
                        if len(marker_list) > 1:
                            for j in range(1, len(marker_list)):
                                if i + j < len(sentence.tokens):
                                    if sentence.tokens[i + j].text != marker_list[j]:
                                        is_equal = False
                                        continue
                        # if it's got found, replace it with a random text from the new list
                        if is_equal and random() < self.p:
                            randi = randint(0, len(new_list[k]) - 1)
                            new = new_list[k][randi].split()
                            #  iterate over the words of the new text
                            for j in range(len(new)):
                                bio_t = token.bio_tag
                                #  if there is the same amount of new and old text, just replace the text etc. of
                                #  the exisiting tokens
                                if j < len(marker_list):
                                    if i + j < len(sentence.tokens):
                                        sentence.tokens[i + j].text = new[j]
                                        sentence.tokens[
                                            i + j
                                        ].pos_tag = tokenmanager.get_pos_tag([new[j]])[
                                            0
                                        ]
                                        sentence.tokens[i + j].bio_tag = bio_t
                                    if j != 0:
                                        pass
                                #  if the new has more words than the old, new tokens must be created
                                if j >= len(marker_list):
                                    tok = model.Token(
                                        text=new[j],
                                        index_in_document=token.index_in_document + j,
                                        pos_tag=tokenmanager.get_pos_tag([new[j]])[0],
                                        bio_tag=bio_t,
                                        sentence_index=token.sentence_index,
                                    )
                                    tokenmanager.create_token(doc, tok, i + j)
                                    i += 1
                                #  when the new has lesser words than the old, the tokens, that are too much must be
                                #  deleted
                                if len(marker_list) > len(new):
                                    num_of_changes = len(marker_list) - len(new)
                                    for i in range(num_of_changes):
                                        tokenmanager.delete_token(
                                            doc, token.index_in_document + len(new)
                                        )

                i += 1
        return doc

    #  creates two separate lists, one with the texts which needs to be replaced and one for the texts which replace
    #  the old one
    def separate(self):
        before = []
        after = []
        marker_to_class = {
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
        class_to_markers = defaultdict(list)
        for k, v in marker_to_class.items():
            class_to_markers[v].append(k.rstrip(","))
        class_to_markers = dict(class_to_markers)
        for m in marker_to_class:
            before.append(m)
            after.append(class_to_markers[marker_to_class[m]])
        return before, after
