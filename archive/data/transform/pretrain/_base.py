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

import abc
from dataclasses import dataclass

from gluonts.dataset import DataEntry


class Transformation(abc.ABC):
    """
    Base class for all transformations.

    This class was based on GluonTS's `Transformation`, but was modified to be
    applied to each time series in a dataset instead of the entire dataset.

    See here for the original implementation:
    https://ts.gluon.ai/stable/api/gluonts/gluonts.transform.html?highlight=transformation#gluonts.transform.Transformation
    """

    @abc.abstractmethod
    def __call__(self, data_entry: DataEntry) -> DataEntry: ...

    def chain(self, other: "Transformation") -> "Chain":
        return Chain([self, other])

    def __add__(self, other: "Transformation") -> "Chain":
        return self.chain(other)

    def __radd__(self, other):
        if other == 0:
            return self
        return other + self


@dataclass
class Chain(Transformation):
    """
    Chain multiple transformations together.

    This class was based on GluonTS's `Chain`, but was modified to be
    applied to each time series in a dataset instead of the entire dataset.

    See here for the original implementation:
    https://ts.gluon.ai/stable/api/gluonts/gluonts.transform.html#gluonts.transform.Chain
    """

    transformations: list[Transformation]

    def __post_init__(self) -> None:
        transformations = []

        for transformation in self.transformations:
            if isinstance(transformation, Identity):
                continue
            elif isinstance(transformation, Chain):
                transformations.extend(transformation.transformations)
            else:
                assert isinstance(transformation, Transformation)
                transformations.append(transformation)

        self.transformations = transformations
        self.__init_passed_kwargs__ = {"transformations": transformations}

    def __call__(self, data_entry: DataEntry) -> DataEntry:
        for t in self.transformations:
            data_entry = t(data_entry)
        return data_entry


class Identity(Transformation):
    """
    Identity transformation that does nothing.

    This class was based on GluonTS's `Identity`, but was modified to be
    applied to each time series in a dataset instead of the entire dataset.

    See here for the original implementation:
    https://ts.gluon.ai/stable/api/gluonts/gluonts.transform.html#gluonts.transform.Identity
    """

    def __call__(self, data_entry: DataEntry) -> DataEntry:
        return data_entry
