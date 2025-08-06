# Forecasting with YingLong Model

## 1. Setup and Imports

# First, we'll install any necessary packages and import all required libraries.
# Install required packages (uncomment if not already installed)

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from dotenv import load_dotenv
from einops import rearrange
from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.itertools import batcher
from gluonts.model import Forecast, evaluate_model
from gluonts.model.forecast import SampleForecast
from gluonts.time_feature import get_seasonality
from tqdm.auto import tqdm

from checkpoints.YingLong_300m.model import GPT
from checkpoints.YingLong_300m.model_config import YingLongConfig
from src.gift_eval.data import Dataset


# 定义日志过滤器以抑制特定的警告信息
class WarningFilter(logging.Filter):
    def __init__(self, text_to_filter):
        super().__init__()
        self.text_to_filter = text_to_filter

    def filter(self, record):
        return self.text_to_filter not in record.getMessage()


gts_logger = logging.getLogger("gluonts.model.forecast")
gts_logger.addFilter(
    WarningFilter("The mean prediction is not stored in the forecast data")
)


# 定义模型配置
@dataclass
class ModelConfig:
    quantile_levels: Optional[List[float]] = None
    forecast_keys: List[str] = field(init=False)
    statsforecast_keys: List[str] = field(init=False)
    intervals: Optional[List[int]] = field(init=False)

    def __post_init__(self):
        self.forecast_keys = ["mean"]
        self.statsforecast_keys = ["mean"]
        if self.quantile_levels is None:
            self.intervals = None
            return

        intervals = set()

        for quantile_level in self.quantile_levels:
            interval = round(200 * (max(quantile_level, 1 - quantile_level) - 0.5))
            intervals.add(interval)
            side = "hi" if quantile_level > 0.5 else "lo"
            self.forecast_keys.append(str(quantile_level))
            self.statsforecast_keys.append(f"{side}-{interval}")

        self.intervals = sorted(intervals)


# 定义 YingLongPredictor 类
class YingLongPredictor:
    def __init__(
        self,
        model,
        prediction_length: int,
        num_samples=20,
        future_token=4096,
    ):
        print("prediction_length:", prediction_length)
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.model = model
        self.future_token = future_token

    def model_predict(
        self,
        context,
        prediction_length,
        future_token,
        scaling=400,
        max_length=4096 * 16,
        *args,
        **predict_kwargs,
    ):
        context = [
            torch.nan_to_num(
                x[-max_length:].to(gpu_device),
                nan=torch.nanmean(x[-max_length:].to(gpu_device)),
            )
            for x in context
        ]

        length = max([len(x) for x in context])
        context = [
            (
                x[-length:]
                if len(x) >= length
                else torch.cat(
                    (torch.ones(length - x.shape[-1]).to(x.device) * torch.mean(x), x)
                )
            )
            for x in context
        ]
        x = torch.stack(context, dim=0)

        # scale_factor = 1
        with torch.no_grad():
            B, _ = x.shape
            logits = 0
            historys = [512, 1024, 2048, 4096]
            if future_token < 1000:
                future_token = (prediction_length // 32 + 1) * 32

            used = 0
            for history in historys:
                if used == 0 or history <= x.shape[-1]:
                    used += 2
                else:
                    continue
                x_train = torch.cat((x.bfloat16(), -x.bfloat16()), dim=0)
                x_train = x_train[..., -history:].bfloat16()

                if x_train.shape[-1] % self.model.patch_size != 0:
                    shape = (
                        x_train.shape[0],
                        self.model.patch_size - x.shape[-1] % self.model.patch_size,
                    )
                    x_train = torch.cat(
                        (
                            torch.ones(shape).to(x_train.device)
                            * x_train.mean(dim=-1, keepdim=True),
                            x_train,
                        ),
                        dim=-1,
                    )
                    x_train = x_train.bfloat16()

                output = self.model(idx=x_train, future_token=future_token)
                if isinstance(output, tuple):
                    logits_all, _ = output
                else:
                    logits_all = output
                logits_all = rearrange(logits_all, "(t b) l c d -> b (l c) d t", t=2)
                logits += logits_all[..., 0] - logits_all[..., 1].flip(dims=[-1])

            logits = logits / used
            sampleHolder = (
                rearrange(logits, "b l c -> b c l")
                .float()
                .contiguous()
                .cpu()
                .detach()[:, :, :prediction_length]
            )
            return torch.nan_to_num(sampleHolder)

    def predict(self, test_data_input, batch_size: int = 1024) -> List[Forecast]:
        predict_kwargs = {"num_samples": self.num_samples}
        while True:
            try:
                forecast_outputs = []
                for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
                    context = [torch.tensor(entry["target"]) for entry in batch]
                    forecast_outputs.append(
                        self.model_predict(
                            context,
                            prediction_length=self.prediction_length,
                            future_token=self.future_token,
                            **predict_kwargs,
                        ).numpy()
                    )
                forecast_outputs = np.concatenate(forecast_outputs)
                break
            except torch.cuda.OutOfMemoryError:
                print(
                    f"OutOfMemoryError at batch_size {batch_size}, reducing to {batch_size // 2}"
                )
                batch_size //= 2

        forecasts = []
        for item, ts in zip(forecast_outputs, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts.append(
                SampleForecast(samples=item, start_date=forecast_start_date)
            )

        return forecasts


device = "cuda:0" if torch.cuda.is_available() else "cpu"
gpu_device = device


def run_task(
    batch_size: int,
    model,
    future_token: int,
    short_tasks: List[str],
    long_tasks: List[str],
    output_dir: str,
    dataset_properties_map: dict,
    model_name: str,
):
    # 实例化评估指标
    metrics = [
        MSE(forecast_type="mean"),
        MSE(forecast_type=0.5),
        MAE(),
        MASE(),
        MAPE(),
        SMAPE(),
        MSIS(),
        RMSE(),
        NRMSE(),
        ND(),
        MeanWeightedSumQuantileLoss(
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ),
    ]

    # 将模型移动到指定设备并设为评估模式
    model = model.to(device).bfloat16()
    model.eval()

    # 构建模型输出目录
    model_name_suffix = f"{model_name.split('/')[-1]}-{future_token}-4096"
    model_output_dir = os.path.join(output_dir, model_name_suffix)
    if not os.path.isdir(model_output_dir):
        os.makedirs(model_output_dir, exist_ok=True)

    # 定义 CSV 文件路径
    csv_file_path = os.path.join(model_output_dir, "all_results.csv")

    # 美化名称映射
    pretty_names = {
        "saugeenday": "saugeen",
        "temperature_rain_with_missing": "temperature_rain",
        "kdd_cup_2018_with_missing": "kdd_cup_2018",
        "car_parts_with_missing": "car_parts",
    }

    # 如果 CSV 文件不存在，创建并写入表头
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "dataset",
                    "model",
                    "eval_metrics/MSE[mean]",
                    "eval_metrics/MSE[0.5]",
                    "eval_metrics/MAE[0.5]",
                    "eval_metrics/MASE[0.5]",
                    "eval_metrics/MAPE[0.5]",
                    "eval_metrics/sMAPE[0.5]",
                    "eval_metrics/MSIS",
                    "eval_metrics/RMSE[mean]",
                    "eval_metrics/NRMSE[mean]",
                    "eval_metrics/ND[0.5]",
                    "eval_metrics/mean_weighted_sum_quantile_loss",
                    "domain",
                    "num_variates",
                ]
            )

    # 合并短期任务和长期任务
    all_datasets = list(set(short_tasks + long_tasks))

    for ds_num, ds_name in enumerate(all_datasets):
        ds_key = ds_name.split("/")[0]
        print(f"Processing dataset: {ds_name} ({ds_num + 1} of {len(all_datasets)})")
        terms = ["short", "medium", "long"]
        for term in terms:
            if (term in ["medium", "long"]) and (ds_name not in long_tasks):
                continue

            if "/" in ds_name:
                ds_key, ds_freq = ds_name.split("/")
                ds_key = ds_key.lower()
                ds_key = pretty_names.get(ds_key, ds_key)
            else:
                ds_key = ds_name.lower()
                ds_key = pretty_names.get(ds_key, ds_key)
                ds_freq = dataset_properties_map[ds_key]["frequency"]
            ds_config = f"{ds_key}/{ds_freq}/{term}"
            print(ds_config)
            to_univariate = (
                False
                if Dataset(name=ds_name, term=term, to_univariate=False).target_dim == 1
                else True
            )
            dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)
            season_length = get_seasonality(dataset.freq)
            print(f"Dataset size: {len(dataset.test_data)}")
            predictor = YingLongPredictor(
                model=model,
                prediction_length=dataset.prediction_length,
                future_token=future_token,
            )
            # 执行模型评估
            res = evaluate_model(
                predictor,
                test_data=dataset.test_data,
                metrics=metrics,
                batch_size=batch_size,
                axis=None,
                mask_invalid_label=True,
                allow_nan_forecast=False,
                seasonality=season_length,
            )

            # 将结果追加到 CSV 文件
            with open(csv_file_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        ds_config,
                        model_name_suffix,
                        res["MSE[mean]"][0],
                        res["MSE[0.5]"][0],
                        res["MAE[0.5]"][0],
                        res["MASE[0.5]"][0],
                        res["MAPE[0.5]"][0],
                        res["sMAPE[0.5]"][0],
                        res["MSIS"][0],
                        res["RMSE[mean]"][0],
                        res["NRMSE[mean]"][0],
                        res["ND[0.5]"][0],
                        res["mean_weighted_sum_quantile_loss"][0],
                        dataset_properties_map[ds_key]["domain"],
                        dataset_properties_map[ds_key]["num_variates"],
                    ]
                )

            print(f"Results for {ds_name} have been written to {csv_file_path}")


# 加载环境变量
load_dotenv()

# 定义模型和输出配置
model_name = "qcw2333/YingLong_300m"
future_token = 4096
output_dir = "results_hf_0"

# 加载模型
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

cfg = YingLongConfig()
model = GPT.from_pretrained(
    "./checkpoints/YingLong_300m",
    torch_dtype=torch.bfloat16,
)
print(type(model))
model = model.to(device).bfloat16()
# model.eval()

# 加载数据集属性
with open("dataset_properties.json", "r") as f:
    dataset_properties_map = json.load(f)


# 定义任务列表，模仿 run-hf-4.sh 的四个并行任务
tasks = [
    {
        "batch_size": 1024,
        "short_tasks": [
            "temperature_rain_with_missing",
            "m4_yearly",
            "electricity/D",
            "restaurant",
            "kdd_cup_2018_with_missing/D",
            "covid_deaths",
            "M_DENSE/D",
            "jena_weather/D",
            "saugeenday/D",
            "saugeenday/W",
            "m4_monthly",
        ],
        "long_tasks": [
            "LOOP_SEATTLE/5T",
            "kdd_cup_2018_with_missing/H",
            "SZ_TAXI/15T",
            "ett1/15T",
            "bizitobs_l2c/5T",
            "bizitobs_l2c/H",
        ],
    },
    {
        "batch_size": 1024,
        "short_tasks": [
            "bitbrains_fast_storage/H",
            "m4_daily",
            "electricity/W",
            "hierarchical_sales/W",
            "m4_weekly",
            "ett2/W",
            "us_births/W",
            "us_births/M",
        ],
        "long_tasks": [
            "bitbrains_rnd/5T",
            "electricity/H",
            "bizitobs_service",
            "jena_weather/H",
            "ett2/15T",
        ],
    },
    {
        "batch_size": 1024,
        "short_tasks": [
            "hospital",
            "LOOP_SEATTLE/D",
            "m4_hourly",
            "ett1/D",
            "ett2/D",
            "ett1/W",
            "saugeenday/M",
        ],
        "long_tasks": [
            "bitbrains_fast_storage/5T",
            "solar/10T",
            "M_DENSE/H",
            "ett1/H",
            "bizitobs_application",
        ],
    },
    {"batch_size": 32, "short_tasks": ["car_parts_with_missing"], "long_tasks": []},
]

# 依次执行所有任务
for idx, task in enumerate(tasks, 1):
    print(f"\n=== Running Task {idx} ===")
    run_task(
        batch_size=task["batch_size"],
        model=model,
        future_token=future_token,
        short_tasks=task["short_tasks"],
        long_tasks=task["long_tasks"],
        output_dir=output_dir,
        dataset_properties_map=dataset_properties_map,
        model_name=model_name,
    )
