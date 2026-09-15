#!/bin/bash

source ./cli/utils.sh
activate_conda_env 

save_every=10

python scripts/gift_eval_pretrain_metadata.py --save-every $save_every