#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import itertools
from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset, Sampler, default_collate, default_convert


@dataclass
class Collate:
    """
    A Callable abstract class for PyTorch DataLoader's collate_fn argument.
    """

    max_length: Optional[int]
    seq_fields: tuple[str, ...]
    pad_func_map: dict[str, Callable[[Sequence[int], np.dtype], np.ndarray]] = field(
        default_factory=dict
    )
    target_field: str = "target"

    def __post_init__(self):
        self.pad_func_map = defaultdict(self._default_pad_func) | self.pad_func_map

    @staticmethod
    def _default_pad_func() -> Callable[[Sequence[int], np.dtype], np.ndarray]:
        return np.zeros

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        raise NotImplementedError


def pad_collate_time_series(batch: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    all_keys = set().union(*batch)

    for key in all_keys:
        values = [sample[key] for sample in batch if key in sample]

        if isinstance(values[0], np.ndarray):
            # Convert to torch tensor
            tensors = [torch.tensor(v) for v in values]

            if tensors[0].ndim == 1:
                # 1D case → shape: (time,)
                padded = pad_sequence(tensors, batch_first=True, padding_value=0.0)
            elif tensors[0].ndim == 2:
                # 2D case → shape: (time, features)
                max_len = max(t.shape[0] for t in tensors)
                padded = torch.stack(
                    [
                        torch.nn.functional.pad(t, (0, 0, 0, max_len - t.shape[0]))
                        for t in tensors
                    ]
                )
            else:
                raise ValueError(
                    f"Unsupported tensor shape {tensors[0].shape} for key '{key}'"
                )

            result[key] = padded

        elif isinstance(values[0], (int, float, str)):
            result[key] = values  # simple fields, no padding

        else:
            raise ValueError(f"Unsupported type for field {key}: {type(values[0])}")

    return result


class PadCollate(Collate):
    """Pads uneven sequences with padding function defined by pad_func_map."""

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        assert all(
            [
                len(sample[self.target_field]) == len(sample[key])
                for sample in batch
                for key in self.seq_fields
            ]
        ), "All fields must have the same length."
        assert all(
            [len(sample[self.target_field]) <= self.max_length for sample in batch]
        ), f"dict[str, torch.Tensor] length must be less than or equal to max_length ({self.max_length})"

        sample_id = self.get_sample_id(batch)
        padded_batch = self.pad_samples(batch)
        merged_batch = padded_batch | dict(sample_id=sample_id)
        return merged_batch

    def pad_samples(
        self, batch: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        for sample in batch:
            length = len(sample[self.target_field])
            for key in self.seq_fields:
                sample[key] = torch.cat(
                    [
                        default_convert(sample[key]),
                        default_convert(
                            self.pad_func_map[key](
                                (self.max_length - length,) + sample[key].shape[1:],
                                sample[key].dtype,
                            )
                        ),
                    ]
                )
        return default_collate(batch)

    def get_sample_id(self, batch: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        sample_id = torch.stack(
            [
                torch.cat([torch.ones(length), torch.zeros(self.max_length - length)])
                for sample in batch
                if (length := len(sample[self.target_field]))
            ]
        ).to(torch.long)
        return sample_id


class PackCollate(Collate):
    """Packs uneven sequences with the first fit decreasing bin packing strategy."""

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        assert all(
            [
                len(sample[self.target_field]) == len(sample[key])
                for sample in batch
                for key in self.seq_fields
            ]
        ), "All fields must have the same length."
        assert all(
            [len(sample[self.target_field]) <= self.max_length for sample in batch]
        ), f"dict[str, torch.Tensor] length must be less than or equal to max_length ({self.max_length})"

        packed_batch, bin_spaces = self.first_fit_decreasing_bin_packing(batch)
        sample_id = self.get_sample_id(packed_batch, bin_spaces)
        merged_batch = self.merge_batch(packed_batch, bin_spaces) | dict(
            sample_id=sample_id
        )
        return merged_batch

    def first_fit_decreasing_bin_packing(
        self,
        batch: list[dict[str, torch.Tensor]],
    ) -> tuple[list[list[dict[str, torch.Tensor]]], np.ndarray]:
        """
        Implements the first fit decreasing bin packing strategy.
        1. Sort the batch by sequence length, long to short.
        2. Initialize an empty list of bins, where each bin has a maximum size of max_length.
        3. Iterate through the sorted batch, inserting samples into the first bin which,
        when the sample is added, does not exceed max_length.

        :param batch: list of samples
        :return:
            - packed_batch - batch which has been packed
            - bin_spaces - length of each bin
        """
        batch = sorted(
            batch, key=lambda sample: len(sample[self.target_field]), reverse=True
        )
        bin_spaces: np.ndarray = np.full(len(batch), self.max_length)
        packed_batch: list[list[dict[str, torch.Tensor]]] = [[]]

        for sample in batch:
            length = len(sample[self.target_field])
            criterion: np.ndarray = bin_spaces - length >= 0
            bin_id: int = criterion.argmax()
            if len(packed_batch) <= bin_id:
                if len(packed_batch) != bin_id:
                    raise ValueError
                packed_batch.append([])

            packed_batch[bin_id].append(sample)
            bin_spaces[bin_id] -= length

        return packed_batch, bin_spaces[: len(packed_batch)]

    def get_sample_id(
        self, batch: list[list[dict[str, torch.Tensor]]], bin_spaces: np.ndarray
    ) -> torch.Tensor:
        """
        Create an array of integers representing the sample id in a sequence.
        dict[str, torch.Tensor] id starts from 1, and 0 represents padding.

        :param batch: packed samples
        :param bin_spaces: length of each bin
        :return: integer array, indicating the sample index given a sequence
        """
        sample_id = torch.stack(
            [
                torch.cat(
                    [
                        torch.ones(len(sample[self.target_field])) * (idx + 1)
                        for idx, sample in enumerate(bin_)
                    ]
                    + [torch.zeros(space)],  # padding
                )
                for bin_, space in zip(batch, bin_spaces)
            ]
        ).to(torch.long)
        return sample_id

    def merge_batch(
        self, batch: list[list[dict[str, torch.Tensor]]], bin_spaces: np.ndarray
    ) -> dict[str, torch.Tensor]:
        """Combines packed samples into dict[str, torch.Tensor] format."""
        batch = {
            key: torch.stack(
                [
                    torch.cat(
                        [default_convert(sample[key]) for sample in bin_]
                        + [
                            default_convert(
                                self.pad_func_map[key](
                                    (space,) + bin_[0][key].shape[1:],
                                    bin_[0][key].dtype,
                                )
                            )
                        ]
                    )
                    for bin_, space in zip(batch, bin_spaces)
                ],
            )
            for key in self.seq_fields
        }
        return batch


@dataclass
class SliceableBatchedSample:
    """A dict[str, torch.Tensor] that can be sliced."""

    data: dict[str, torch.Tensor]

    def __post_init__(self):
        assert all(
            [
                len(self.data[key]) == len(self.data[next(iter(self.data))])
                for key in self.data.keys()
            ]
        )

    def __len__(self) -> int:
        return len(self.data[next(iter(self.data))])

    def __getitem__(self, item: slice) -> "SliceableBatchedSample":
        return SliceableBatchedSample(
            {key: self.data[key][item] for key in self.data.keys()}
        )


class Metadata(NamedTuple):
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass
class BatchedSampleQueue:
    """
    Queue data structure storing batched samples.

    :param container: internal queue data structure storing batched samples
    :param schema: format specification for batched samples
    """

    container: deque[SliceableBatchedSample] = field(default_factory=deque)
    schema: Optional[dict[str, Metadata]] = None

    def _check_schema(self, batch: SliceableBatchedSample):
        """
        Ensure that all samples in the batch follows the required schema.
        If a schema has not been specified, then all samples in the batch should have the same schema.
        """
        if self.schema is None:
            self.schema = {
                key: Metadata(
                    shape=tuple(batch.data[key].shape[1:]), dtype=batch.data[key].dtype
                )
                for key in batch.data.keys()
            }
        else:
            assert all(
                [
                    (key in batch.data)
                    and (metadata.shape == tuple(batch.data[key].shape[1:]))
                    and (metadata.dtype == batch.data[key].dtype)
                    for key, metadata in self.schema.items()
                ]
            ), "batch must have the same schema as the first batch"

    def append(self, batch: SliceableBatchedSample | dict[str, torch.Tensor]):
        """Appends a batch to the end of the queue."""
        if not isinstance(batch, SliceableBatchedSample):
            batch = SliceableBatchedSample(batch)
        self._check_schema(batch)
        self.container.append(batch)

    def appendleft(self, batch: SliceableBatchedSample | dict[str, torch.Tensor]):
        """Appends a batch to the start of the queue."""
        if not isinstance(batch, SliceableBatchedSample):
            batch = SliceableBatchedSample(batch)
        self._check_schema(batch)
        self.container.appendleft(batch)

    def popleft(self, size: int) -> dict[str, torch.Tensor]:
        """Pops a batch from the start of the queue."""
        if size > len(self):
            raise ValueError(
                f"pop size ({size}) must be less than or equal to queue size ({len(self)})"
            )

        out = BatchedSampleQueue()
        while len(out) < size:
            curr = self.container.popleft()
            if len(out) + len(curr) > size:
                self.appendleft(curr[size - len(out) :])
                curr = curr[: size - len(out)]
            out.append(curr)
        return out.as_batched_data()

    def as_batched_data(self) -> dict[str, torch.Tensor]:
        """Returns the queue as a dict[str, torch.Tensor]"""
        return {
            key: torch.cat([batch.data[key] for batch in self.container], dim=0)
            for key in self.schema.keys()
        }

    def __len__(self) -> int:
        """Total number of samples in the queue."""
        return sum(len(batch) for batch in self.container)


@dataclass
class _BatchedSampleIterator:
    """
    Iterator returning batched samples with a fixed batch size.

    :param dataloader_iter: iterator returning batched samples, may not be with fixed batch size
    :param batch_size: the batch size of batched samples to return
    :param drop_last: whether to drop the last batch of it does not have batch_size samples
    :param fill_last: whether to fill the last batch with padding if it does not have batch_size samples
    :param pad_func_map: mapping to padding functions
    """

    dataloader_iter: Iterator[dict[str, torch.Tensor]]
    batch_size: int
    drop_last: bool
    fill_last: bool

    def __post_init__(self):
        self.queue = BatchedSampleQueue()

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        while (data := self._next_batch()) is None:
            continue
        return data

    def _next_batch(self) -> Optional[dict[str, torch.Tensor]]:
        """
        :return: either None or the next batch of samples
        :raises StopIteration: if the queue is empty
        """
        if len(self.queue) < self.batch_size:
            # check if there are sufficient samples in the queue
            # if not, extract the next batch from dataloader_iter and return None
            try:
                data = next(self.dataloader_iter)
                self.queue.append(data)
                return None
            except StopIteration:
                # no more batches from the dataloader_iter
                # check for drop_last and fill_last strategy
                if self.drop_last or len(self.queue) == 0:
                    raise StopIteration
                elif self.fill_last:
                    self._pad_queue(self.batch_size - len(self.queue))

        batch = self.queue.popleft(min(self.batch_size, len(self.queue)))
        return batch

    def _pad_queue(self, size: int):
        if self.queue.schema is None:
            raise ValueError("schema must be set before padding")
        padding = {}
        for key, metadata in self.queue.schema.items():
            pad_shape = (size,) + metadata.shape
            pad_array = np.zeros(pad_shape, dtype=np.float32)
            pad_tensor = default_convert(pad_array).to(metadata.dtype)
            padding[key] = pad_tensor
        self.queue.append(padding)

    def has_next(self) -> bool:
        """Check if iterator still has next."""
        if len(self.queue) < self.batch_size:
            try:
                next_batch = next(self)
                self.queue.appendleft(next_batch)
            except StopIteration:
                return False
        return True


class DataLoader:
    """
    Wrapper on PyTorch's DataLoader class implementing:
    - packing
    - number of batches per epoch
    - cycle
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        batch_size_factor: float = 1.0,
        cycle: bool = False,
        num_batches_per_epoch: Optional[int] = None,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        num_workers: int = 0,
        collate_fn: Optional[Collate] = None,
        pin_memory: bool = False,
        drop_last: bool = True,
        fill_last: bool = False,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ):
        """
        Only wrapper specific arguments are documented below.
        See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        for documentation on PyTorch DataLoader arguments.

        :param batch_size_factor: multiply the batch_size given to PyTorch's DataLoader.
        :param cycle: whether to cycle dataloader infinitely.
        :param num_batches_per_epoch: number of batches per epoch.
        :param fill_last: whether to fill the last batch with padding.
        """
        if num_batches_per_epoch is not None:
            assert cycle, "can only set 'num_batches_per_epoch' when 'cycle=True'"

        self.dataloader = TorchDataLoader(
            dataset=dataset,
            batch_size=int(batch_size * batch_size_factor),
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers and num_workers > 0,
        )
        self.batch_size = batch_size
        self.cycle = cycle
        self.num_batches_per_epoch = num_batches_per_epoch
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.fill_last = fill_last
        self.iterator: Optional[_BatchedSampleIterator] = None

    def __iter__(self) -> Iterator:
        if self.iterator is None or not self.iterator.has_next():
            dataloader_iter = (
                iter(self.dataloader)
                if not self.cycle
                else itertools.chain.from_iterable(itertools.repeat(self.dataloader))
            )
            self.iterator = _BatchedSampleIterator(
                dataloader_iter=dataloader_iter,
                batch_size=self.batch_size,
                drop_last=self.drop_last,
                fill_last=self.fill_last,
            )
        return itertools.islice(self.iterator, self.num_batches_per_epoch)

    @property
    def worker_init_fn(self) -> Optional[Callable[[int], None]]:
        """Getter for worker_init_fn"""
        return self.dataloader.worker_init_fn

    @worker_init_fn.setter
    def worker_init_fn(self, worker_init_fn: Optional[Callable[[int], None]]):
        """Setter for worker_init_fn"""
        self.dataloader.worker_init_fn = worker_init_fn
