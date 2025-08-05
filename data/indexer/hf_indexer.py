from collections.abc import Iterable
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from datasets import Dataset
from datasets.features import Sequence
from datasets.formatting import query_table

from ._base import Indexer


class HFDatasetIndexer(Indexer):
    """
    Indexer for Hugging Face Datasets. Allows for fast access to the underlying
    data.

    This class was created by uni2ts. See here for the original implementation:
    https://github.com/SalesforceAIResearch/uni2ts/blob/main/src/uni2ts/data/indexer/hf_dataset_indexer.py
    """

    def __init__(self, dataset: Dataset, metadata: dict, uniform: bool = False):
        """
        Initialize the indexer with a Hugging Face Dataset.

        Args:
            dataset (Dataset): The Hugging Face Dataset to index.
            metadata (dict): Metadata associated with the dataset.
            uniform (bool): Whether the underlying data has uniform length.
        """
        super().__init__(uniform=uniform)
        self.dataset = dataset
        self.metadata = metadata
        self.features = dict(self.dataset.features)
        self.non_seq_cols = [
            name
            for name, feat in self.features.items()
            if not isinstance(feat, Sequence)
        ]
        self.seq_cols = [
            name for name, feat in self.features.items() if isinstance(feat, Sequence)
        ]
        self.dataset.set_format("numpy", columns=self.non_seq_cols)

    def __len__(self) -> int:
        return len(self.dataset)

    def _getitem_int(self, idx: int) -> dict[str, Any]:
        non_seqs = self.dataset[idx]
        pa_subtable = query_table(self.dataset.data, idx, indices=self.dataset._indices)
        seqs = {
            col: self._pa_column_to_numpy(pa_subtable, col)[0] for col in self.seq_cols
        }
        return non_seqs | seqs

    def _getitem_iterable(self, idx: Iterable[int]) -> dict[str, Any]:
        non_seqs = self.dataset[idx]
        pa_subtable = query_table(self.dataset.data, idx, indices=self.dataset._indices)
        seqs = {
            col: self._pa_column_to_numpy(pa_subtable, col) for col in self.seq_cols
        }
        return non_seqs | seqs

    def _getitem_slice(self, idx: slice) -> dict[str, Any]:
        non_seqs = self.dataset[idx]
        pa_subtable = query_table(self.dataset.data, idx, indices=self.dataset._indices)
        seqs = {
            col: self._pa_column_to_numpy(pa_subtable, col) for col in self.seq_cols
        }
        return non_seqs | seqs

    def _pa_column_to_numpy(
        self, pa_table: pa.Table, column_name: str
    ) -> list[Any] | list[Any]:
        pa_array: pa.Array = pa_table.column(column_name)
        feature = self.features[column_name]

        if isinstance(pa_array, pa.ChunkedArray):
            if isinstance(feature.feature, Sequence):
                array = [
                    flat_slice.flatten().to_numpy(False).reshape(feat_length, -1)
                    for chunk in pa_array.chunks
                    for i in range(len(chunk))
                    if (flat_slice := chunk.slice(i, 1).flatten())
                    and (
                        feat_length := (
                            feature.length if feature.length != -1 else len(flat_slice)
                        )
                    )
                ]
            else:
                array = [
                    chunk.slice(i, 1).flatten().to_numpy(False)
                    for chunk in pa_array.chunks
                    for i in range(len(chunk))
                ]
        elif isinstance(pa_array, pa.ListArray):
            if isinstance(feature.feature, Sequence):
                flat_slice = pa_array.flatten()
                feat_length = (
                    feature.length if feature.length != -1 else len(flat_slice)
                )
                array = [flat_slice.flatten().to_numpy(False).reshape(feat_length, -1)]
            else:
                array = [pa_array.flatten().to_numpy(False)]
        else:
            raise NotImplementedError

        return array

    def get_proportional_probabilities(self, field: str = "target") -> np.ndarray:
        """
        Obtain proportion of each time series based on number of time steps.
        Leverages pyarrow.compute for fast implementation.

        :param field: field name to measure time series length
        :return: proportional probabilities
        """

        if self.uniform:
            return self.get_uniform_probabilities()

        if self[0]["target"].ndim > 1:
            lengths = pc.list_value_length(
                pc.list_flatten(pc.list_slice(self.dataset.data.column(field), 0, 1))
            )
        else:
            lengths = pc.list_value_length(self.dataset.data.column(field))
        lengths = lengths.to_numpy()
        probs = lengths / lengths.sum()
        return probs
