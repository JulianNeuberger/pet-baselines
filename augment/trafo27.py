import typing

from augment import base_abbreviation_trafo
from data import model


class Trafo27Step(base_abbreviation_trafo.BaseAbbreviationStep):
    def __init__(
        self, dataset: typing.List[model.Document]
    ):
        abbreviations = self._load()
        super().__init__(dataset, abbreviations)

    @staticmethod
    def _load() -> typing.Dict[str, str]:
        abbreviations = {}
        with open(f"./resources/abbreviations/27.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                key, value = line.split(":", 1)
                abbreviations[key] = value
        return abbreviations
