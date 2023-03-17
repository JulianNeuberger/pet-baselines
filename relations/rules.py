import typing

import data


class RelationExtractionRule:
    def get_relations(self, document: data.Document) -> typing.List[data.Relation]:
        raise NotImplementedError()

    @staticmethod
    def get_next_index_of_mention_with_tag(document: data.Document, start_index: int, tags: typing.List[str],
                                           search_backwards: bool = False) -> typing.Optional[int]:
        search_indices = range(start_index, len(document.mentions))
        if search_backwards:
            search_indices = range(start_index, -1, -1)

        mentions = document.mentions.copy()
        for i in search_indices:
            mention = mentions[i]
            if mention.ner_tag in tags:
                return i
        return None


class SequenceFlowsRule(RelationExtractionRule):
    def __init__(self, triggering_elements: typing.List[str], target_tag: str):
        self._elements = triggering_elements
        self._tag = target_tag

    def get_relations(self, document: data.Document) -> typing.List[data.Relation]:
        relations = []
        for i, current_mention in enumerate(document.mentions):
            cur_mention_behavioural = current_mention.ner_tag in self._elements
            if not cur_mention_behavioural:
                continue

            next_behavioral_element_index = self.get_next_index_of_mention_with_tag(document, i + 1, self._elements)

            if next_behavioral_element_index is None:
                continue

            flow_relation = data.Relation(
                head_entity_index=i,
                tail_entity_index=next_behavioral_element_index,
                tag=self._tag
            )

            if document.contains_relation(flow_relation):
                print(f'[{self.__class__.__name__}] '
                      f'{flow_relation.pretty_print(document)} already in relations, not adding.')
                continue

            relations.append(flow_relation)

        return relations


class SameGatewayRule(RelationExtractionRule):
    def __init__(self, triggering_elements: typing.List[str], target_tag: str):
        self._elements = triggering_elements
        self._tag = target_tag

    def get_relations(self, document: data.Document) -> typing.List[data.Relation]:
        relations = []
        for i, mention in enumerate(document.mentions):
            if mention.ner_tag not in self._elements:
                continue

            next_gateway_index = self.get_next_index_of_mention_with_tag(document, i + 1, [mention.ner_tag])
            if next_gateway_index is None:
                continue

            next_gateway = document.mentions[next_gateway_index]

            distance = next_gateway.sentence_index - mention.sentence_index
            if distance > 1:
                continue

            relations.append(data.Relation(
                head_entity_index=i,
                tail_entity_index=next_gateway_index,
                tag=self._tag
            ))

        return relations


class GatewayActivityRule(RelationExtractionRule):
    def __init__(self, gateway_tags: typing.List[str], activity_tag: str, same_gateway_tag: str, flow_tag: str):
        self._gateways = gateway_tags
        self._activity = activity_tag
        self._same_gateway = same_gateway_tag
        self._tag = flow_tag

    def get_relations(self, document: data.Document) -> typing.List[data.Relation]:
        relations = []
        for i, mention in enumerate(document.mentions):
            if mention.ner_tag not in self._gateways:
                continue
            if self.is_mention_part_of_relation_with_tag(document, i, self._same_gateway):
                continue

            next_activity_index = self.get_next_index_of_mention_with_tag(document, i + 1, [self._activity])
            if next_activity_index is None:
                continue

            flow_relation = data.Relation(
                head_entity_index=i,
                tail_entity_index=next_activity_index,
                tag=self._tag
            )

            if document.contains_relation(flow_relation):
                print(f'[{self.__class__.__name__}] {flow_relation} already in relations, not adding.')
                continue

            relations.append(flow_relation)
        return relations

    @staticmethod
    def is_mention_part_of_relation_with_tag(document: data.Document, mention_index: int, tag: str) -> bool:
        for relation in document.relations:
            if relation.tag != tag:
                continue
            if relation.tail_entity_index == mention_index:
                return True
            if relation.head_entity_index == mention_index:
                return True

        return False


class ActorPerformerRecipientRule(RelationExtractionRule):
    def __init__(self, actor_tag: str, activity_tag: str, performer_tag: str, recipient_tag: str):
        self._actor = actor_tag
        self._activity = activity_tag
        self._performer = performer_tag
        self._recipient = recipient_tag

    def get_relations(self, document: data.Document) -> typing.List[data.Relation]:
        relations = []

        for i, mention in enumerate(document.mentions):
            if not mention.ner_tag == self._activity:
                continue

            performer_index = self.get_next_index_of_mention_with_tag(document, i, [self._actor],
                                                                      search_backwards=True)
            if performer_index is not None:
                if document.mentions[performer_index].sentence_index == mention.sentence_index:
                    relations.append(data.Relation(
                        head_entity_index=i,
                        tail_entity_index=performer_index,
                        tag=self._performer
                    ))

            recipient_index = self.get_next_index_of_mention_with_tag(document, i, [self._actor])
            if recipient_index is not None:
                if document.mentions[recipient_index].sentence_index == mention.sentence_index:
                    relations.append(data.Relation(
                        head_entity_index=i,
                        tail_entity_index=recipient_index,
                        tag=self._recipient
                    ))

        return relations


class FurtherSpecificationRule(RelationExtractionRule):
    def __init__(self, further_specification_element_tag: str, activity_tag: str,
                 further_specification_relation_tag: str):
        self._further_spec = further_specification_element_tag
        self._activity = activity_tag
        self._tag = further_specification_relation_tag

    def get_relations(self, document: data.Document) -> typing.List[data.Relation]:
        relations = []

        for i, mention in enumerate(document.mentions):
            if mention.ner_tag != self._further_spec:
                continue

            left_index = self.get_next_index_of_mention_with_tag(document, i, [self._activity], search_backwards=True)
            right_index = self.get_next_index_of_mention_with_tag(document, i, [self._activity])

            if left_index is None and right_index is None:
                continue

            chosen_index: int
            if left_index is None:
                chosen_index = right_index
            elif right_index is None:
                chosen_index = left_index
            else:
                # both not None, choose nearer
                left_candidate = document.mentions[left_index]
                right_candidate = document.mentions[right_index]

                # closest token of left candidate is the right-most
                left_end = document.sentences[left_candidate.sentence_index].tokens[left_candidate.token_indices[-1]].index_in_document
                # closest token of right candidate is the left-most
                right_start = document.sentences[right_candidate.sentence_index].tokens[right_candidate.token_indices[0]].index_in_document

                source_start = document.sentences[mention.sentence_index].tokens[min(mention.token_indices)].index_in_document
                source_end = document.sentences[mention.sentence_index].tokens[max(mention.token_indices)].index_in_document

                distance_to_left = source_start - left_end
                distance_to_right = right_start - source_end
                if distance_to_left <= distance_to_right:
                    chosen_index = left_index
                else:
                    chosen_index = right_index

            relations.append(data.Relation(
                head_entity_index=chosen_index,
                tail_entity_index=i,
                tag=self._tag
            ))

        return relations


class UsesRelationRule(RelationExtractionRule):
    def __init__(self, activity_data_tag: str, activity_tag: str, uses_relation_tag: str):
        self._activity_data = activity_data_tag
        self._activity = activity_tag
        self._tag = uses_relation_tag

    def get_relations(self, document: data.Document) -> typing.List[data.Relation]:
        relations = []
        for i, mention in enumerate(document.mentions):
            if mention.ner_tag != self._activity_data:
                continue

            activity_index = self.get_next_index_of_mention_with_tag(document, i - 1,
                                                                     [self._activity], search_backwards=True)

            if activity_index is None or document.mentions[activity_index].sentence_index != mention.sentence_index:
                activity_index = self.get_next_index_of_mention_with_tag(document, i + 1, [self._activity])

            if activity_index is None or document.mentions[activity_index].sentence_index != mention.sentence_index:
                continue

            r = data.Relation(
                head_entity_index=activity_index,
                tail_entity_index=i,
                tag=self._tag
            )
            relations.append(r)

        return relations
