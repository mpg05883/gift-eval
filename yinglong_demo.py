import torch

from checkpoints.YingLong_300m.model import GPT
from checkpoints.YingLong_300m.model_config import YingLongConfig

cfg = YingLongConfig()
model = GPT.from_pretrained(
    "./checkpoints/YingLong_300m",
    torch_dtype=torch.bfloat16,
).to("cuda")

# prepare input
batch_size, lookback_length = 1, 2880
seqs = torch.randn(batch_size, lookback_length).bfloat16().cuda()

# generate forecast
prediction_length = 96
output = model.generate(seqs, future_token=prediction_length)

print(output.shape)
