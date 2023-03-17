import dataclasses

import typing


@dataclasses.dataclass
class Document:
    text: str
    sentences: typing.List['Sentence'] = dataclasses.field(default_factory=list)
    mentions: typing.List['Mention'] = dataclasses.field(default_factory=list)
    entities: typing.List['Entity'] = dataclasses.field(default_factory=list)
    relations: typing.List['Relation'] = dataclasses.field(default_factory=list)

    def contains_relation(self, relation: 'Relation') -> bool:
        return relation.to_tuple(self) in [e.to_tuple(self) for e in self.relations]

    def copy(self) -> 'Document':
        return Document(
            text=self.text,
            sentences=[s.copy() for s in self.sentences],
            mentions=[m.copy() for m in self.mentions],
            relations=[r.copy() for r in self.relations]
        )

    def get_mentions_by_token_index(self, token_index: int) -> typing.Optional['Mention']:
        for mention in self.mentions:
            if token_index in mention.token_indices:
                return mention
        return None

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

    def to_tuple(self, *args) -> typing.Tuple:
        return (self.ner_tag.lower(), self.sentence_index) + tuple(self.token_indices)

    def text(self, document: Document):
        return ' '.join([document.sentences[self.sentence_index].tokens[i].text for i in self.token_indices])

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

    def to_tuple(self, *args) -> typing.Tuple:
        return set(self.mention_indices),

    def copy(self) -> 'Entity':
        return Entity(
            mention_indices=[i for i in self.mention_indices]
        )


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
        assert type(self.head_entity_index) == int, f'head, {self}'
        assert type(self.tail_entity_index) == int, f'tail, {self}'
        return (document.mentions[self.head_entity_index].to_tuple(),
                self.tag.lower(),
                document.mentions[self.tail_entity_index].to_tuple())

    def pretty_print(self, document: Document):
        head_mention = document.mentions[self.head_entity_index]
        tail_mention = document.mentions[self.tail_entity_index]

        return f'[{head_mention.pretty_print(document)}]--[{self.tag}]-->[{tail_mention.pretty_print(document)}]'


@dataclasses.dataclass
class Token:
    text: str
    index_in_document: int
    pos_tag: str
    bio_tag: str

    def copy(self) -> 'Token':
        return Token(
            text=self.text,
            index_in_document=self.index_in_document,
            pos_tag=self.pos_tag,
            bio_tag=self.bio_tag
        )