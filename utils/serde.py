import pandas as pd
import torch
from gluonts.torch.distributions import AffineTransformed
from gluonts.torch.distributions.studentT import StudentT
from gluonts.torch.model.forecast import DistributionForecast
from gluonts.model import SampleForecast


def serialize_tensor(tensor: torch.tensor) -> float | list:
    """
    Serializes a PyTorch tensor to a Python dictionary.

    Args:
        tensor (torch.tensor): A PyTorch tensor that can be of any shape.

    Returns:
        float | list: A serialized representation of the tensor, which can be a
        single float if the tensor has one element, or a list of floats if the
        tensor has multiple elements.
    """
    # TODO: Change this so it can serialize tensor into Hydra-compatible format
    return tensor.item() if len(tensor) == 1 else tensor.detach().cpu().tolist()


def deserialize_tensor(data: float | list) -> torch.tensor:
    """
    Deserializes a Python dictionary back into a PyTorch tensor.

    Args:
        data (float | list): A serialized representation of a tensor, which can
            be a single float or a list of floats.

    Returns:
        torch.tensor | torch.tensor: A PyTorch tensor created from the input
            data.
    """
    # TODO: Change this so it can deserialize data from Hydra-compatible format
    return torch.tensor(data)


def serialize_student_t(student_t: StudentT) -> dict[str, float | list]:
    """
    Serializes a Student's T distribution to a Python dictionary.

    See here for GluonTS's StudentT documentation:
    https://ts.gluon.ai/dev/_modules/gluonts/torch/distributions/studentT.html#StudentT

    Args:
        student_t (StudentT): A StudentT distribution object that contains
            the degrees of freedom (df), location (loc), and scale parameters.

    Returns:
        dict[str, float | list]: A dictionary representation of the Student's T
            distribution containing the df, loc, and scale parameters.
    """
    # TODO: Change this so it can serialize StudentT into Hydra-compatible format
    return {
        "type": student_t.__class__.__name__,
        "args": {
            "df": serialize_tensor(student_t.df),
            "loc": serialize_tensor(student_t.loc),
            "scale": serialize_tensor(student_t.scale),
        },
    }


def deserialize_student_t(data: dict[str, float | list]) -> StudentT:
    """
    Deserializes a Python dictionary to Student's T distribution.

    See here for GluonTS's StudentT documentation:
    https://ts.gluon.ai/dev/_modules/gluonts/torch/distributions/studentT.html#StudentT

    Args:
        data (dict[str, float | list]): A dictionary representation of a
            Student's T distribution, where the values are either single floats
            or lists of floats.

    Returns:
        StudentT: A StudentT distribution object created from the input data.
    """
    # TODO: Change this so it can deserialize data from Hydra-compatible format
    return StudentT(
        df=deserialize_tensor(data["args"]["df"]),
        loc=deserialize_tensor(data["args"]["loc"]),
        scale=deserialize_tensor(data["args"]["scale"]),
    )


def serialize_affine_transformed(
    affine_transformed: AffineTransformed,
) -> dict[str, float | list]:
    """
    Serializes an affine transformation to a Python dictionary. Assumes the
    affine transformation's base distribution is a Student's T distribution.

    See here for GluonTS's AffineTransformed documentation:
    https://ts.gluon.ai/dev/_modules/gluonts/torch/distributions/affine_transformed.html#AffineTransformed

    Args:
        affine_transformed (AffineTransformed): An AffineTransformed object
            that contains a StudentT object.

    Returns:
        dict[str, float | list]: A dictionary representation of the affine
            transformation containing the loc and scale parameters.
    """
    # TODO: Change this so it can serialize affine_transformed into Hydra-compatible format
    return {
        "type": affine_transformed.__class__.__name__,
        "args": {
            "base_dist": serialize_student_t(affine_transformed.base_dist),
            "loc": serialize_tensor(affine_transformed.loc),
            "scale": serialize_tensor(affine_transformed.scale),
        },
    }


def deserialize_affine_transformed(
    data: dict[str, float | list],
) -> AffineTransformed:
    """
    Deserializes a Python dictionary to an affine transformation. Assumes the
    affine transformation's base distribution is a Student's T distribution.

    See here for GluonTS's AffineTransformed documentation:
    https://ts.gluon.ai/dev/_modules/gluonts/torch/distributions/affine_transformed.html#AffineTransformed

    Args:
        data (dict[str, float | list]): A dictionary representation of an
            affine transformation, where the values are either single floats or
            lists of floats.

    Returns:
        AffineTransformed: An AffineTransformed object created from the
            input data.
    """
    # TODO: Change this so it can serialize affine_transformed into Hydra-compatible format
    return AffineTransformed(
        base_distribution=deserialize_student_t(data["args"]["base_dist"]),
        loc=deserialize_tensor(data["args"]["loc"]),
        scale=deserialize_tensor(data["args"]["scale"]),
    )


def serialize_forecasts(
    forecasts: list[DistributionForecast],
) -> dict[str, list]:
    """
    Serializes a list of DistributionForecast objects to a Python dictionary.
    Assumes all DistributionForecast objects have affine transformation
    distributions and that the affine transformation's base distribution is a
    Student's T distribution.

    See here for GluonTS's DistributionForecast documentation:
    https://ts.gluon.ai/dev/_modules/gluonts/torch/model/forecast.html#DistributionForecast

    Args:
        forecasts (list[DistributionForecast]): A list of DistributionForecast
            objects to be serialized. Each DistributionForecast represents a
            single model's test set forecast on each time series in a dataset.

    Returns:
        dict[str, list]: A dictionary containing the serialized data from the
            DistributionForecast objects. Each key corresponds to a specific
            attribute of the DistributionForecast objects, and the values are
            lists containing each forecast's attributes.
    """
    # TODO: Change this so it can serialize forecasts into Hydra-compatible format
    if isinstance(forecasts[0], SampleForecast):
        return {
            "samples": [forecast.samples.tolist() for forecast in forecasts],
            "start_date": [str(forecast.start_date) for forecast in forecasts],
            "freq": [forecast.freq for forecast in forecasts],
            "item_id": [forecast.item_id for forecast in forecasts],
            "info": [forecast.info for forecast in forecasts],
        }
    
    return {
        "distribution": [
            serialize_affine_transformed(forecast.distribution)
            for forecast in forecasts
        ],
        "start_date": [str(forecast.start_date) for forecast in forecasts],
        "freq": [forecast.freq for forecast in forecasts],
        "item_id": [forecast.item_id for forecast in forecasts],
        "info": [forecast.info for forecast in forecasts],
    }


def deserialize_forecasts(
    data: dict[str, list],
) -> list[DistributionForecast]:
    """
    Deserializes a Python dictionary to a list of DistributionForecast objects.

    NOTE: Assumes all Forecast objects have Student's T distributions.

    See here for GluonTS's DistributionForecast documentation:
    https://ts.gluon.ai/dev/_modules/gluonts/torch/model/forecast.html#DistributionForecast

    Args:
        data (dict[str, list]): A dictionary representation of Forecast
            objects, where each key corresponds to a specific attribute of the
            Forecast objects and the values are lists containing each forecast's
            attributes.

    Returns:
        list[DistributionForecast]: A list of DistributionForecast objects
        created from the input data.
    """
    # TODO: Change this so it can deserialize forecasts from Hydra-compatible format
    forecasts = []
    for i, _ in enumerate(data["distribution"]):
        forecast = DistributionForecast(
            distribution=deserialize_affine_transformed(data["distribution"][i]),
            start_date=pd.Period(data["start_date"][i], freq=data["freq"][i]),
            item_id=data["item_id"][i],
            info=data["info"][i],
        )
        forecasts.append(forecast)
    return forecasts
