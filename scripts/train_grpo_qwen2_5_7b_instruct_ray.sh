#!/bin/bash

set -x

source /home/duhe/miniconda3/etc/profile.d/conda.sh
conda activate openrlhf

cd /cpfs01/user/duhe/OpenRLHF

./examples/scripts/train_grpo_qwen2_5_7b_instruct_ray.sh
