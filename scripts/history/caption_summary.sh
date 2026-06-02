#!/bin/zsh

source ~/.zshrc
conda activate lavis
cd ./Documents/competition/BC-AVSMod
export PYTHONPATH="./Documents/competition/BC-AVSMod:$PYTHONPATH"

# CUDA_VISIBLE_DEVICES=7 nohup python LLaMA/caption_summary.py > logs/captions/10299.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python LLaMA/caption_summary.py > logs/captions/14334.log 2>&1 &

