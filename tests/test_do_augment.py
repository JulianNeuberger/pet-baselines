import inspect
import typing

from augment import base, params
from data import model


def collect_all_trafos(
    base_class: typing.Type,
) -> typing.List[typing.Type[base.AugmentationStep]]:
    sub_classes = []

    immediate_sub_classes = base_class.__subclasses__()
    immediate_sub_classes = [
        c for c in immediate_sub_classes if not inspect.isabstract(c)
    ]
    sub_classes.extend(immediate_sub_classes)
    for sub_class in immediate_sub_classes:
        child_sub_classes = collect_all_trafos(sub_class)
        sub_classes.extend(child_sub_classes)

    return sub_classes


def get_test_doc():
    sentences = [
        model.Sentence(
            tokens=[
                model.Token(
                    text="First",
                    index_in_document=0,
                    pos_tag="JJ",
                    bio_tag="",
                    sentence_index=0,
                ),
                model.Token(
                    text="sentence",
                    index_in_document=1,
                    pos_tag="NN",
                    bio_tag="",
                    sentence_index=0,
                ),
                model.Token(
                    text="content",
                    index_in_document=2,
                    pos_tag="NN",
                    bio_tag="",
                    sentence_index=0,
                ),
                model.Token(
                    text=".",
                    index_in_document=3,
                    pos_tag=".",
                    bio_tag="",
                    sentence_index=0,
                ),
            ]
        ),
        model.Sentence(
            tokens=[
                model.Token(
                    text="Second",
                    index_in_document=4,
                    pos_tag="JJ",
                    bio_tag="",
                    sentence_index=1,
                ),
                model.Token(
                    text="short",
                    index_in_document=5,
                    pos_tag="JJ",
                    bio_tag="",
                    sentence_index=1,
                ),
                model.Token(
                    text="sentence",
                    index_in_document=6,
                    pos_tag="NN",
                    bio_tag="",
                    sentence_index=1,
                ),
                model.Token(
                    text="!",
                    index_in_document=7,
                    pos_tag=".",
                    bio_tag="",
                    sentence_index=1,
                ),
            ]
        ),
        model.Sentence(
            tokens=[
                model.Token(
                    text="Third",
                    index_in_document=8,
                    pos_tag="NNP",
                    bio_tag="",
                    sentence_index=2,
                ),
                model.Token(
                    text="and",
                    index_in_document=9,
                    pos_tag="CC",
                    bio_tag="",
                    sentence_index=2,
                ),
                model.Token(
                    text="last",
                    index_in_document=10,
                    pos_tag="JJ",
                    bio_tag="",
                    sentence_index=2,
                ),
                model.Token(
                    text="!",
                    index_in_document=11,
                    pos_tag=".",
                    bio_tag="",
                    sentence_index=2,
                ),
            ]
        ),
    ]

    mentions = [
        model.Mention(ner_tag="A", sentence_index=0, token_indices=[1]),
        model.Mention(ner_tag="B", sentence_index=1, token_indices=[0, 1, 2]),
    ]

    entities = [model.Entity(mention_indices=[0]), model.Entity(mention_indices=[1])]

    relations = [
        model.Relation(
            head_entity_index=0, tail_entity_index=1, tag="RR", evidence=[0, 1]
        )
    ]

    document = model.Document(
        name="",
        sentences=sentences,
        mentions=mentions,
        entities=entities,
        relations=relations,
        text="",
    )
    return document


def test_do_augment():
    trafo_classes = collect_all_trafos(base.AugmentationStep)
    print(trafo_classes)
    for clazz in trafo_classes:
        print(f"Testing {clazz.__name__}...")
        doc = get_test_doc()
        args = {"dataset": [doc]}
        for param in clazz.get_params():
            if isinstance(param, params.NumberParam):
                args[param.name] = param.min_value
            elif isinstance(param, params.ChoiceParam):
                args[param.name] = param.choices[0]
                if param.max_num_picks > 1:
                    args[param.name] = [param.choices[0]]
            elif isinstance(param, params.BooleanParameter):
                args[param.name] = True
        trafo = clazz(**args)
        trafo.do_augment(doc)
