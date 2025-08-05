from enum import Enum, StrEnum, auto


class SplitType(StrEnum):
    """
    Represents which part of split GIFT-Eval is being used (pretraining or
    train-test)

    Attributes:
        PRETRAIN: Indicates the pretraining phase.
        TRAIN_TEST: Indicates the the train-test/finetuning phase.
    """

    PRETRAIN = "pretrain"
    TRAIN_TEST = "train_test"


class Term(StrEnum):
    """
    Represents the forecasting horizon category.

    Attributes:
        SHORT: Short-term forecasting.
        MEDIUM: Medium-term forecasting.
        LONG: Long-term forecasting.

    Properties:
        multipler (int): A factor used to scale the prediction length based
            on the term.
    """

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


class ForecasterType(Enum):
    """
    Represents the scope of the forecaster being trained.

    Attributes:
        TERM: Trained on all datasets of a specific term (e.g., all short-term
        datasets).
        DOMAIN: Trained on all datasets within a single domain.
        DATASET: Trained on a single dataset.
    """

    TERM = 0
    DOMAIN = auto()
    DATASET = auto()


class Domain(StrEnum):
    """
    Represents the dataset's domain.

    Attributes:
        CLIMATE: Datasets related to weather, climate, or environmental
            monitoring.
        CLOUDOPS: Datasets related to cloud infrastructure and operations.
        ECON_FIN: Economic and financial datasets.
        HEALTHCARE: Datasets from the healthcare domain.
        NATURE: Scientific or biological datasets.
        SALES: Datasets tracking retail or product sales.
        TRANSPORT: Datasets involving traffic or transportation.
        WEB: Datasets from web or online platforms.
        WEB_CLOUDOPS: A combined or hybrid domain covering both Web and
            CloudOps.
    """

    CLIMATE = "Climate"
    CLOUDOPS = "CloudOps"
    ECON_FIN = "Econ/Fin"
    HEALTHCARE = "Healthcare"
    NATURE = "Nature"
    SALES = "Sales"
    TRANSPORT = "Transport"
    WEB = "Web"
    WEB_CLOUDOPS = "Web/CloudOps"
