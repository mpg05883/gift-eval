#!/bin/bash

# source ./cli/utils.sh
# activate_conda_env 

save_every=5

uv run python scripts/gift_eval_pretrain_metadata.py --save-every $save_every