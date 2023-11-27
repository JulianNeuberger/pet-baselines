import abc
import inspect

from augment import params
import typing

from data import model


class AugmentationStep(abc.ABC):
    def __init__(self, dataset: typing.List[model.Document], **kwargs):
        self.dataset = dataset

    def do_augment(self, doc: model.Document) -> model.Document:
        raise NotImplementedError()

    @staticmethod
    def validate_params(clazz: typing.Type["AugmentationStep"]):
        args = inspect.getfullargspec(clazz.__init__).args
        missing_args = []
        for param in clazz.get_params():
            found_arg = False
            for arg in args:
                if param.name == arg:
                    found_arg = True
                    break
            if not found_arg:
                missing_args.append(param)
        if len(missing_args) > 0:
            raise TypeError(f"Missing arguments in __init__ method of {clazz.__name__}: {missing_args}")

    @staticmethod
    def get_params() -> typing.List[typing.Union[params.Param]]:
        raise NotImplementedError()

    @staticmethod
    def get_sequences(
        doc: model.Document,
    ) -> typing.List[typing.List[model.Token]]:
        """
        Returns a list of sequences (lists) of tokens, that have
        the same ner tag.
        """
        tagged_sequences = []

        for sentence in doc.sentences:
            last_sequence: typing.List[model.Token] = []
            last_mention: typing.Optional[model.Mention] = None
            for token in sentence.tokens:
                mentions = doc.get_mentions_for_token(token)
                if len(mentions) == 0:
                    current_mention = None
                else:
                    current_mention = mentions[0]
                if current_mention != last_mention:
                    if len(last_sequence) > 0:
                        tagged_sequences.append(last_sequence)
                    last_sequence = []
                last_mention = current_mention
                last_sequence.append(token)
            if len(last_sequence) > 0:
                tagged_sequences.append(last_sequence)

        return tagged_sequences
