from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

# Replace with your desired path
save_path = Path("./checkpoints/YingLong_300m")
save_path.mkdir(parents=True, exist_ok=True)
save_path = str(save_path)

# Download and save model + tokenizer to that path
model = AutoModelForCausalLM.from_pretrained(
    "qcw2333/YingLong_300m", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "qcw2333/YingLong_300m", trust_remote_code=True
)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
