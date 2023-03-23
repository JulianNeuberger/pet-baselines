import dataclasses

import typing


@dataclasses.dataclass
class Document:
    text: str
    name: str
    sentences: typing.List['Sentence'] = dataclasses.field(default_factory=list)
    mentions: typing.List['Mention'] = dataclasses.field(default_factory=list)
    entities: typing.List['Entity'] = dataclasses.field(default_factory=list)
    relations: typing.List['Relation'] = dataclasses.field(default_factory=list)

    def contains_relation(self, relation: 'Relation') -> bool:
        return relation.to_tuple(self) in [e.to_tuple(self) for e in self.relations]

    def contains_entity(self, entity: 'Entity') -> bool:
        return entity.to_tuple(self) in [e.to_tuple(self) for e in self.entities]

    def entity_index_for_mention(self, mention: 'Mention') -> int:
        mentions_as_tuples = [m.to_tuple(self) for m in self.mentions]
        mention_index = mentions_as_tuples.index(mention.to_tuple(self))
        for i, e in enumerate(self.entities):
            if mention_index in e.mention_indices:
                return i
        print(mention_index)
        print(self.entities)
        raise ValueError(f'Document contains no entity using mention {mention}, '
                         f'which should not happen, but can happen, '
                         f'if entities are not properly resolved')

    def copy(self,
             clear_mentions: bool = False,
             clear_relations: bool = False,
             clear_entities: bool = False) -> 'Document':
        return Document(
            name=self.name,
            text=self.text,
            sentences=[s.copy() for s in self.sentences],
            mentions=[] if clear_mentions else [m.copy() for m in self.mentions],
            relations=[] if clear_relations else [r.copy() for r in self.relations],
            entities=[] if clear_entities else [e.copy() for e in self.entities]
        )

    @property
    def tokens(self):
        ret = []
        for s in self.sentences:
            ret.extend(s.tokens)
        return ret


@dataclasses.dataclass
class Sentence:
    tokens: typing.List['Token'] = dataclasses.field(default_factory=list)

    @property
    def num_tokens(self):
        return len(self.tokens)

    def copy(self) -> 'Sentence':
        return Sentence([t.copy() for t in self.tokens])


@dataclasses.dataclass
class Mention:
    ner_tag: str
    sentence_index: int
    token_indices: typing.List[int] = dataclasses.field(default_factory=list)

    def get_tokens(self, document: Document) -> typing.List['Token']:
        return [document.sentences[self.sentence_index].tokens[i] for i in self.token_indices]

    def to_tuple(self, *args) -> typing.Tuple:
        return (self.ner_tag.lower(), self.sentence_index) + tuple(self.token_indices)

    def text(self, document: Document):
        return ' '.join([t.text for t in self.get_tokens(document)])

    def contains_token(self, token: 'Token', document: 'Document') -> bool:
        if token.sentence_index != self.sentence_index:
            return False
        for own_token_idx in self.token_indices:
            own_token = document.sentences[self.sentence_index].tokens[own_token_idx]
            if own_token.index_in_document == token.index_in_document:
                return True
        return False

    def pretty_print(self, document: Document):
        return f'{self.text(document)} ({self.ner_tag}, s{self.sentence_index}:{min(self.token_indices)}-{max(self.token_indices)})'

    def copy(self) -> 'Mention':
        return Mention(
            ner_tag=self.ner_tag,
            sentence_index=self.sentence_index,
            token_indices=[i for i in self.token_indices]
        )


@dataclasses.dataclass
class Entity:
    mention_indices: typing.List[int] = dataclasses.field(default_factory=list)

    def to_tuple(self, document: Document) -> typing.Tuple:
        mentions = [document.mentions[i] for i in self.mention_indices]

        return frozenset([m.to_tuple(document) for m in mentions]),

    def copy(self) -> 'Entity':
        return Entity(
            mention_indices=[i for i in self.mention_indices]
        )

    def pretty_print(self, document: Document):
        return f'Entity [{[document.mentions[m].pretty_print(document) for m in self.mention_indices]}]'


@dataclasses.dataclass
class Relation:
    head_entity_index: int
    tail_entity_index: int
    tag: str

    def copy(self) -> 'Relation':
        return Relation(
            head_entity_index=self.head_entity_index,
            tail_entity_index=self.tail_entity_index,
            tag=self.tag
        )

    def to_tuple(self, document: Document) -> typing.Tuple:
        return (self.tag.lower(),
                document.entities[self.head_entity_index].to_tuple(document),
                document.entities[self.tail_entity_index].to_tuple(document))

    def pretty_print(self, document: Document):
        head_entity = document.entities[self.head_entity_index]
        tail_entity = document.entities[self.tail_entity_index]

        head_mention = document.mentions[head_entity.mention_indices[0]]
        tail_mention = document.mentions[tail_entity.mention_indices[0]]

        return f'[{head_mention.pretty_print(document)} (+{len(head_entity.mention_indices) - 1})]' \
               f'--[{self.tag}]-->' \
               f'[{tail_mention.pretty_print(document)} (+{len(tail_entity.mention_indices) - 1})]'


@dataclasses.dataclass
class Token:
    text: str
    index_in_document: int
    pos_tag: str
    bio_tag: str
    sentence_index: int

    def copy(self) -> 'Token':
        return Token(
            text=self.text,
            index_in_document=self.index_in_document,
            pos_tag=self.pos_tag,
            bio_tag=self.bio_tag,
            sentence_index=self.sentence_index
        )
