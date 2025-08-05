from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, List

from gluonts.dataset import DataEntry

from ._base import Transformation


@dataclass
class SetValue:
    value: Any

    def __call__(self, data_entry: DataEntry) -> Any:
        return self.value


@dataclass
class LambdaSetFieldIfNotPresent(Transformation):
    field: str
    get_value: Callable[[DataEntry], Any]

    @staticmethod
    def set_field(data_entry: DataEntry, field: str, value: Any) -> DataEntry:
        if field not in data_entry.keys():
            data_entry[field] = value
        return data_entry

    def __call__(self, data_entry: DataEntry) -> DataEntry:
        return self.set_field(data_entry, self.field, self.get_value(data_entry))


@dataclass
class SelectFields(Transformation):
    """
    Only keeps the listed fields in the data entry. If `allow_missing` is True,
    it will not raise an error if some fields are missing from the data entry.

    This transformation is based on GluonTS's `SelectFields`, but modified to
    be applied to each time series in a dataset instead of the entire dataset.

    See here for the original implementation:
    https://ts.gluon.ai/stable/api/gluonts/gluonts.transform.html?highlight=selectfields#gluonts.transform.SelectFields
    """

    fields: list[str]
    allow_missing: bool = False

    def __call__(self, data_entry: DataEntry) -> DataEntry:
        if self.allow_missing:
            return {f: data_entry[f] for f in self.fields if f in data_entry}
        return {f: data_entry[f] for f in self.fields}


@dataclass
class RemoveFields(Transformation):
    """
    Remove specified fields from the data entry.

    This transformation is based on GluonTS's `RemoveFields`, but modified to
    be applied to each time series in a dataset instead of the entire dataset.

    See here for the original implementation:
    https://ts.gluon.ai/stable/api/gluonts/gluonts.transform.html?highlight=removefields#gluonts.transform.RemoveFields
    """

    fields: List[str]

    def __call__(self, data_entry: DataEntry) -> DataEntry:
        data_entry = {
            key: value for key, value in data_entry.items() if key not in self.fields
        }
        return data_entry
