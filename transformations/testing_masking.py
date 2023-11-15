import copy
import typing
from random import random

from transformers import pipeline
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import data
from data import model
from transformations import tokenmanager

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


def do_augment2(doc2: model.Document) -> model.Document:
    doc = copy.deepcopy(doc2)
    i = 0
    p = 1
    while i < len(doc.sentences):
        if random() > p:
            sent_length = len(doc.sentences[i].tokens) - 1
            if i < len(doc.sentences) - 1:
                # transfer Mentions
                for mention in doc.mentions:
                    if mention.sentence_index == i + 1:
                        mention.sentence_index -= 1
                        for ment_id in mention.token_indices:
                            ment_id += sent_length
                # transfer Tokens
                tok_arr = []
                for j, token in enumerate(doc.sentences[i + 1].tokens):
                    tok = model.Token(token.text, token.index_in_document - 1, token.pos_tag, token.bio_tag,
                                      token.sentence_index - 1)
                    tok_arr.append(tok)
                # delete sentence
                tokenmanager.delete_sentence(doc, i + 1)

                for j, tok in enumerate(tok_arr):
                    tokenmanager.create_token(doc, tok, sent_length + j)

                # delete punct
                tokenmanager.delete_token(doc, tok_arr[-1].index_in_document + 1)
        i += 1
    return doc




model = M2M100ForConditionalGeneration.from_pretrained(
            "facebook/m2m100_418M"
        )
tokenizer = M2M100Tokenizer.from_pretrained(
            "facebook/m2m100_418M"
        )
src_lang = "en"
pivot_lang = "de"
p = 1


def do_augment(doc):
    doc = doc.copy()
    for sentence in doc.sentences:
        i = 0
        while i < len(sentence.tokens) - 1:
            token = sentence.tokens[i]
            current_bio = tokenmanager.get_bio_tag_short(token.bio_tag)
            text_before = ""
            j = 0
            while (
                    tokenmanager.get_bio_tag_short(sentence.tokens[i + j].bio_tag)
                    == current_bio
            ):
                if i + j < len(sentence.tokens) - 1:
                    if j == 0:
                        text_before += sentence.tokens[i + j].text
                    else:
                        text_before += " "
                        text_before += sentence.tokens[i + j].text
                j += 1
                if i + j > len(sentence.tokens) - 1:
                    break

            if random() >= p:
                i += j
                continue
            if text_before in [",", ".", "?", "!", ":", "#", "-"]:
                translated = text_before
            else:
                translated = back_translate(en=text_before)
            text_before_list = text_before.split()
            translated_list = translated
            diff = len(translated_list) - len(text_before_list)
            if diff > 0:
                token.text = translated_list[0]
                token.pos_tag = tokenmanager.get_pos_tag([token.text])[0]
                if len(translated_list) > 1:
                    for k in range(1, len(translated_list)):
                        tok = model.Token(
                            text=translated_list[k],
                            index_in_document=token.index_in_document + i + k,
                            pos_tag=tokenmanager.get_pos_tag([token.text])[0],
                            bio_tag=tokenmanager.get_bio_tag_based_on_left_token(
                                token.bio_tag
                            ),
                            sentence_index=token.sentence_index,
                        )
                        tokenmanager.create_token(doc, tok, i + k)
            elif diff == 0:
                for k in range(0, len(translated_list)):
                    index_in_doc = token.index_in_document
                    sentence.tokens[i + k].text = translated_list[k]
                    sentence.tokens[i + k].pos_tag = tokenmanager.get_pos_tag(
                        [token.text]
                    )[0]
            else:
                for k in range(len(translated_list)):
                    sentence.tokens[i + k].text = translated_list[k]
                    sentence.tokens[i + k].pos_tag = tokenmanager.get_pos_tag(
                        [token.text]
                    )[0]
                for k in range(len(translated_list), len(text_before_list)):
                    tokenmanager.delete_token(
                        doc,
                        sentence.tokens[i + len(translated_list)].index_in_document,
                    )
            i = i + j + diff
    return doc


#  returns the back translated text, when it's not working, it returns the old text
def back_translate(en: str):
    try:
        de = translate(en, pivot_lang, model, tokenizer)
        en_new = translate(de, src_lang, model, tokenizer)
    except Exception as ex:
        print(ex)
        print("Returning Default due to Run Time Exception")
        en_new = en
    return en_new


def translate(sentence, target_lang, model, tokenizer):
    tokenizer.src_lang = src_lang
    encoded_source_sentence = tokenizer(sentence, return_tensors="pt")
    generated_target_tokens = model.generate(
        **encoded_source_sentence,
        forced_bos_token_id=tokenizer.get_lang_id(target_lang)
    )
    target_sentence = tokenizer.batch_decode(
        generated_target_tokens, skip_special_tokens=True
    )
    return target_sentence

#tokenmanager.delete_sentence(docs[0], 0)
print(translate(sentence="the nice professor", target_lang="de", model=model, tokenizer=tokenizer)[0].split())
#print(do_augment(docs[0]))