import typing
from random import randint, random
from augment import base
from data import model
from transformations import tokenmanager


class Trafo82Step(base.AugmentationStep):

    def __init__(self, short_to_long=True, long_to_short=True, p=1, bank=1):
        self.short_to_long = short_to_long
        self.long_to_short = long_to_short
        self.bank = bank
        self.p = p

    def do_augment(self, doc: model.Document):
        #  when Bank is 0, take the Bank from Trafo 82, the Bank is always split in contracted and expanded list
        if self.bank == 0:
            abb_list = Trafo82Step.load(self, "abb.txt")
            contracted_list = Trafo82Step.separate_into_contracted_and_expanded_form(self, abb_list, True)[0]
            expanded_list = Trafo82Step.separate_into_contracted_and_expanded_form(self, abb_list, True)[1]
        #  when Bank is 1, take the Bank from 110
        elif self.bank == 1:
            abb_list = Trafo82Step.load_bank110(self)
            contracted_list = abb_list[1]
            expanded_list = abb_list[0]
        #  when Bank is >= 2, take the Bank from 27
        else:
            abb_list = Trafo82Step.load(self, "abb27.txt")
            contracted_list = Trafo82Step.separate_into_contracted_and_expanded_form(self, abb_list, True)[0]
            expanded_list = Trafo82Step.separate_into_contracted_and_expanded_form(self, abb_list, True)[1]
        token_text_sentence = []
        index_in_doc_blacklist = []
        # search for contracted Abbreviation and replace with the expanded form
        if self.short_to_long and random() < self.p:
            for sentence in doc.sentences:
                index_in_sentence = 0
                for token in sentence.tokens:
                    index_in_sentence += 1
                    counter_abb = 0
                    all_abb = []
                    # identify the place of the Abbreviation in the Bank and put all possible expanded forms in a list
                    for abb in contracted_list:
                        counter_abb += 1
                        if abb == token.text:
                            all_abb.append(counter_abb)
                    #  when there is an expanded form, replace the token with the new one and create new ones
                    if all_abb != []:
                        #  select a random expanded form from the list and split in tokens
                        rand_int = randint(0, len(all_abb) - 1)
                        expanded_list_splitted = expanded_list[all_abb[rand_int] - 1].split()
                        token_list = []
                        #  put the token indices of the abbreviations we expanded in a blacklist, to not contract them
                        #  again in the next step
                        for i in range(len(expanded_list_splitted)):
                            index_in_doc_blacklist.append(token.index_in_document + i)
                        #  for the first expanded token, we adapt the already existing token
                        #  and if there are any further, create a token list, store them in and create new tokens
                        mention_id = tokenmanager.get_mentions(doc, index_in_sentence, token.sentence_index)
                        token.text = expanded_list_splitted[0]
                        token.pos_tag = tokenmanager.get_pos_tag([token.text])[0]
                        for i in range(1, len(expanded_list_splitted)):
                            token_list.append(model.Token(text=expanded_list_splitted[i],
                                                          index_in_document=token.index_in_document + i,
                                                          pos_tag=
                                                          tokenmanager.get_pos_tag([expanded_list_splitted[i]])[0],
                                                          bio_tag=tokenmanager.get_bio_tag_based_on_left_token(
                                                              token.bio_tag), sentence_index=token.sentence_index))
                        for j in range(len(token_list)):
                            if mention_id == []:
                                tokenmanager.create_token(doc, token_list[j],
                                                          index_in_sentence=index_in_sentence + j)
                            else:
                                tokenmanager.create_token(doc, token_list[j],
                                                          index_in_sentence=index_in_sentence + j,
                                                          mention_index=mention_id[0])
        # search for expanded abbreviations
        if self.long_to_short and random() < self.p:
            expanded_double_list = []
            #  create a list of the expanded Abbreviations which are word for word stored as a list, --> 2D list
            for m in range(len(expanded_list)):
                expanded_double_list.append(expanded_list[m].split())
            #  iterate over the list of expanded abbreviations
            for i in range(len(expanded_double_list)):
                text = expanded_double_list[i][0]
                for sentence in doc.sentences:
                    k = 0
                    while k < len(sentence.tokens):
                        tok_text = sentence.tokens[k].text
                        #  when the first word of the expanded abbreviation matches the token text, look if the
                        if tok_text == text:
                            is_equal = True
                            if len(expanded_double_list[i]) > 1:
                                for j in range(1, len(expanded_double_list[i])):
                                    if k + j < len(sentence.tokens):
                                        if sentence.tokens[k + j].text != expanded_double_list[i][j]:
                                            is_equal = False
                            if is_equal and sentence.tokens[k].index_in_document not in index_in_doc_blacklist:
                                # if is_equal:
                                # print(contracted_list[i])
                                sentence.tokens[k].text = contracted_list[i]
                                sentence.tokens[k].pos_tag = tokenmanager.get_pos_tag([contracted_list[i]])[0]
                                if len(expanded_double_list[i]) > 1:
                                    for j in range(1, len(expanded_double_list[i])):
                                        if k + j < len(sentence.tokens):
                                            if k == len(sentence.tokens) - 1 and sentence.tokens[k + j].text == ".":
                                                pass
                                            else:
                                                tokenmanager.delete_token(doc, sentence.tokens[k + j].index_in_document)
                        k += 1
        return doc

    def load(self, bank) -> typing.List[typing.List[str]]:
        """
        Load from a file, the list of abbreviations as tuple(list) of the expanded and the contracted form.

        Parameters:
            path_to_file: Path to file containing the abbreviations.
                The file should contains one abbreviation pair per line, separated by a semicolon. e.g.: ACCT:account.

        Returns:
            List of pairs of contracted and expanded abbreviations.
        """
        with open(
                f"./transformations/trafo82/{bank}"
                ,
                "r", encoding='utf-8'
        ) as fd:
            content = fd.read()
            line = content.split("\n")

        if line[-1] == "":
            line = line[:-1]

        return [element.split(":", 1) for element in line]

    def load_bank110(self):
        sep = "\t"
        temp_acronyms = []
        contracted = []
        expanded = []
        with open("./transformations/trafo82/acronyms.tsv", "r", encoding="utf-8") as file:
            for line in file:
                key, value = line.strip().split(sep)
                #temp_acronyms[key] = value
                contracted.append(key)
                expanded.append(value)
        # Place long keys first to prevent overlapping
        acronyms = {}
        for k in sorted(temp_acronyms, key=len, reverse=True):
            acronyms[k] = temp_acronyms[k]
        acronyms = acronyms
        return contracted, expanded
    def separate_into_contracted_and_expanded_form(self,
                                                   abbreviations: typing.List, case_sensitive: bool
                                                   ) -> typing.Tuple[typing.List, typing.List]:
        """
        Split a given list of abbreviations pairs into two lists.
        One for contracted form and the other for expanded form.
        The abbreviation list has the form ACCT:account.

        Parameters:
            abbreviations: A list of pairs of contracted and expanded abbreviations.
            case_sensitive: If we want to check abbreviations while being case sensitive.

        Returns:
            A Tuple of two lists.
        """
        contracted_abbreviations = []
        expanded_abbreviations = []

        for contracted_form, expanded_form in abbreviations:
            if case_sensitive:
                contracted_abbreviations.append(contracted_form)
                expanded_abbreviations.append(expanded_form)
            else:
                contracted_abbreviations.append(contracted_form.lower())
                expanded_abbreviations.append(expanded_form.lower())
        return contracted_abbreviations, expanded_abbreviations
