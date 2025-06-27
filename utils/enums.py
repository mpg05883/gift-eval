from enum import Enum, StrEnum, auto


class SplitType(StrEnum):
    PRETRAIN = "pretrain"
    TRAIN_TEST = "train_test"


class Term(StrEnum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

    @property
    def multipler(self) -> int:
        return {
            Term.SHORT: 1,
            Term.MEDIUM: 10,
            Term.LONG: 15,
        }[self]


class ModelType(Enum):
    GENERAL = 0
    DOMAIN = auto()
    DATASET = auto()


class Domain(StrEnum):
    CLIMATE = "Climate"
    CLOUDOPS = "CloudOps"
    ECON_FIN = "Econ/Fin"
    HEALTHCARE = "Healthcare"
    NATURE = "Nature"
    SALES = "Sales"
    TRANSPORT = "Transport"
    WEB = "Web"
    WEB_CLOUDOPS = "Web/CloudOps"
