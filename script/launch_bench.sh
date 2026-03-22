#!/bin/bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export HF_HOME="/root/workspace/jyj/workspaces/MultiBD/.hf_home"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_METRICS_CACHE="$HF_HOME/metrics"
export CUDA_VISIBLE_DEVICES=0
export HF_ALLOW_CODE_EVAL=1
export TRITON_PRINT_AUTOTUNING=1
export DEBUG_WITH_ONE_REQ=0

# Use :- for default; a single : is substring expansion and slashes become "division" in arithmetic.
CONFIG_FILE="${CONFIG_FILE:-diffulex_bench/configs/example.yml}"

echo "Running DiffulexBench using $CONFIG_FILE"

python -m diffulex_bench.main --config "$CONFIG_FILE"