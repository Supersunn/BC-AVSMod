#!/bin/zsh

source ~/.zshrc
conda activate msdd
cd ./AVE/CLIPAdapter/
export PYTHONPATH="./AVE/CLIPAdapter/:$PYTHONPATH"

tensorboard --logdir ./AVE/CLIPAdapter/checkpoints/AVE/FinedCLIP_gpu0/2025-10-31@06-13-38/0
#=======================================================================================================================================================================
# FinedCLIP

--20250123  --is_fully_task 
CUDA_VISIBLE_DEVICES=0 nohup python clipprad/run.py --net_type FinedCLIP --dataset_name AVE --batch_size 8 --lr 5e-5 --adapter_mlp_ratio 32 --is_fully_task > logs/AVE/20250123_fully_woamp_epoch=8_maxepoch=15_lr=5e-5_ra=32_h=4_loss=ce.log 2>&1
CUDA_VISIBLE_DEVICES=7 nohup python clipprad/run.py --net_type FinedCLIP --dataset_name AVE --batch_size 8 --lr 5e-5 --adapter_mlp_ratio 32 > logs/AVE/20250123_weakly_woamp_epoch=8_maxepoch=15_lr=5e-5_ra=32_h=4_loss=ce.log 2>&1
CUDA_VISIBLE_DEVICES=5 nohup python clipprad/run.py --net_type FinedCLIP --dataset_name AVE --batch_size 8 --lr 5e-5 --adapter_mlp_ratio 32 --is_fully_task > logs/AVE/20250224_fully_woamp_epoch=8_maxepoch=15_lr=5e-5_ra=32_h=4_loss=ce.log 2>&1
CUDA_VISIBLE_DEVICES=3 nohup python clipprad/run.py --net_type FinedCLIP --dataset_name AVE --batch_size 8 --lr 5e-5 --adapter_mlp_ratio 32 --is_fully_task > logs/AVE/20250224_moe_mamba_fully_woamp_epoch=8_maxepoch=15_lr=5e-5_ra=32_h=4_loss=ce.log 2>&1
-----------------------------------



CUDA_VISIBLE_DEVICES=2 nohup python clipprad/run.py --dataset_name AVE --batch_size 8 --lr 5e-5 --adapter_mlp_ratio 32 > logs/AVE/20250108_bs=8_lr=5e-5_ra=32.log 2>&1 &

Hyper-parameter[2025.10.30] 
(1) epoch
[10epoch]--lr 5e-5 --adapter_mlp_ratio 32   alpha=0.5
CUDA_VISIBLE_DEVICES=0 nohup python clipprad/run.py --dataset_name AVE --batch_size 8 --lr 5e-5 --adapter_mlp_ratio 32 > logs/AVE/20250830_wall_woamp_epoch=8_ra=32_5e-5.log 2>&1 &
[15epoch]--lr 5e-5 --adapter_mlp_ratio 32   alpha=0.5
CUDA_VISIBLE_DEVICES=1 nohup python clipprad/run.py --dataset_name AVE --batch_size 8 --lr 5e-5 --adapter_mlp_ratio 32 > logs/AVE/20250830_wall_woamp_maxepoch=15_epoch=8_ra=32_5e-5.log 2>&1 &
[20epoch]
CUDA_VISIBLE_DEVICES=0 nohup python clipprad/run.py --dataset_name AVE --batch_size 8 --lr 5e-5 --adapter_mlp_ratio 32 > logs/AVE/20250830_wall_woamp_maxepoch=20_epoch=8_ra=32_5e-5.log 2>&1 &
(2) lr
[epoch=15]
CUDA_VISIBLE_DEVICES=1 nohup python clipprad/run.py --dataset_name AVE --batch_size 8 --lr 5e-4 --adapter_mlp_ratio 32 > logs/AVE/20250830_wall_woamp_maxepoch=15_epoch=8_ra=32_5e-4.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python clipprad/run.py --dataset_name AVE --batch_size 8 --lr 3e-3 --adapter_mlp_ratio 32 > logs/AVE/20250830_wall_woamp_maxepoch=15_epoch=8_ra=32_3e-3.log 2>&1 &
--lr 6e-6
(3) adapter_mlp_ratio
16
64
(4) alpha
0.2
CUDA_VISIBLE_DEVICES=1 nohup python clipprad/run.py --dataset_name AVE --batch_size 8 --lr 5e-5 --adapter_mlp_ratio 32 > logs/AVE/20251101_wall_woamp_maxepoch=15_epoch=8_ra=32_5e-5_alpha=0.2.log 2>&1 &
0.7
CUDA_VISIBLE_DEVICES=2 nohup python clipprad/run.py --dataset_name AVE --batch_size 8 --lr 5e-5 --adapter_mlp_ratio 32 > logs/AVE/20251101_wall_woamp_maxepoch=15_epoch=8_ra=32_5e-5_alpha=0.7.log 2>&1 &
(5) head
4
128
CUDA_VISIBLE_DEVICES=2 nohup python clipprad/run.py --dataset_name AVE --batch_size 8 --lr 5e-5 --adapter_mlp_ratio 32 > logs/AVE/20251101_woamp_epoch=8_maxepoch=15_lr=5e-5_ra=32_alpha=0.5_h=128_loss=ce_mul.log 2>&1 &

(6) Loss   ./train.py  L38
LceLmul
CUDA_VISIBLE_DEVICES=3 nohup python clipprad/run.py --dataset_name AVE --batch_size 8 --lr 5e-5 --adapter_mlp_ratio 32 > logs/AVE/20251101_wall_woamp_maxepoch=15_epoch=8_ra=32_5e-5_alpha=0.5_wLceLmul.log 2>&1 &







[Ablation Experiment]
tensorboard --logdir /home/sunchao/AVE/CLIPAdapter24/checkpoints/AVE/FinedCLIP_gpu0/2025-10-26@07-25-44/0
http://localhost:6006/
epoch number: /home/sunchao/AVE/CLIPAdapter24/clipprad/configs/config.py:  Lin78, "max_epoch": 20
open AudioCLIP parameters: /home/sunchao/AVE/CLIPAdapter24/clipprad/src/models/model.py: L410, m.requires_grad = True/False
open Fusion: /home/sunchao/AVE/CLIPAdapter24/clipprad/src/models/model.py: L569
open Caption: /home/sunchao/AVE/CLIPAdapter24/clipprad/src/models/model.py: L566
open Adapter: /home/sunchao/AVE/CLIPAdapter24/clipprad/src/models/model.py: L511-L523
# /home/sunchao/AVE/CLIPAdapter24/clipprad/src/models/submodules/nets/clip/model.py: L341, class CLIPEncoderLayer(nn.Module):
mod alpha: /home/sunchao/AVE/CLIPAdapter/clipprad/configs/config.py L81
mod head: /home/sunchao/AVE/CLIPAdapter/clipprad/src/models/model.py L569

1、Loss
CUDA_VISIBLE_DEVICES=0 nohup python clipprad/run.py --dataset_name AVE --batch_size 8 --lr 5e-5 --adapter_mlp_ratio 32 > logs/AVE/20251102_woamp_epoch=8_maxepoch=15_lr=5e-5_ra=32_alpha=0.7__h=4_loss=ce_mul.log 2>&1 &



2、woVbranch  /home/sunchao/AVE/CLIPAdapter/clipprad/src/models/fusion_model.py
CUDA_VISIBLE_DEVICES=4 nohup python clipprad/run.py --dataset_name AVE --batch_size 8 --lr 5e-5 > logs/AVE/20250813_xr_woadapt_5e-5_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python clipprad/run.py --dataset_name AVE --batch_size 8 --lr 5e-5 > logs/AVE/20250814_xr_woadapt_sce_5e-5.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python clipprad/run.py --dataset_name AVE --batch_size 10 --lr 5e-5 --is_amp --adapter_mlp_ratio 4 > logs/AVE/20250815_xr_wall_woadapter_ra=4_5e-5.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python clipprad/run.py --dataset_name AVE --batch_size 8 --lr 5e-5 --adapter_mlp_ratio 32 > logs/AVE/20250816_xr_wall_woadapter_ra=32_5e-5.log 2>&1 &



3、woCIC、SNS  /home/sunchao/AVE/CLIPAdapter/clipprad/src/models/fusion_model.py
CUDA_VISIBLE_DEVICES=2 nohup python clipprad/run.py --dataset_name AVE --batch_size 8 --lr 5e-5 > logs/AVE/20250813_xr_womtf_5e-5_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python clipprad/run.py --dataset_name AVE --batch_size 8 --lr 5e-5 > logs/AVE/20250814_xr_womtf_sce_5e-5.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python clipprad/run.py --dataset_name AVE --batch_size 10 --lr 5e-5 --is_amp --adapter_mlp_ratio 4 > logs/AVE/20250815_xr_wall_womtf_ra=4_5e-5.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python clipprad/run.py --dataset_name AVE --batch_size 8 --lr 5e-5 --adapter_mlp_ratio 32 > logs/AVE/20250816_xr_wall_womtf_ra=32_5e-5.log 2>&1 &





6、Hyper-parameter
[=8]
[=2]--adapter_mlp_ratio
CUDA_VISIBLE_DEVICES=5 nohup python clipprad/run.py --dataset_name AVE --batch_size 8 --lr 5e-5 --adapter_mlp_ratio 2 > logs/AVE/20250814_xr_waudio_wmtf_wadapt_wprompt_sce_rio=2_5e-5.log 2>&1 &
[=4]--adapter_mlp_ratio
CUDA_VISIBLE_DEVICES=6 nohup python clipprad/run.py --dataset_name AVE --batch_size 8 --lr 5e-5 --adapter_mlp_ratio 4 > logs/AVE/20250814_xr_waudio_wmtf_wadapt_wprompt_sce_rio=4_5e-5.log 2>&1 &
[=6]--adapter_mlp_ratio
CUDA_VISIBLE_DEVICES=7 nohup python clipprad/run.py --dataset_name AVE --batch_size 8 --lr 5e-5 --adapter_mlp_ratio 6 > logs/AVE/20250814_xr_waudio_wmtf_wadapt_wprompt_sce_rio=6_5e-5.log 2>&1 &
[=10]--adapter_mlp_ratio
CUDA_VISIBLE_DEVICES=1 nohup python clipprad/run.py --dataset_name AVE --batch_size 8 --lr 5e-5 --adapter_mlp_ratio 10 > logs/AVE/20250814_xr_waudio_wmtf_wadapt_wprompt_sce_rio=10_5e-5.log 2>&1 &
[=12]--adapter_mlp_ratio
CUDA_VISIBLE_DEVICES=4 nohup python clipprad/run.py --dataset_name AVE --batch_size 8 --lr 5e-5 --adapter_mlp_ratio 12 > logs/AVE/20250814_xr_waudio_wmtf_wadapt_wprompt_sce_rio=12_5e-5.log 2>&1 &

[2]
CUDA_VISIBLE_DEVICES=1 nohup python clipprad/run.py --dataset_name AVE --batch_size 10 --lr 5e-5 --is_amp --adapter_mlp_ratio 2 > logs/AVE/20250816_xr_wall_wamp_epoch=10_ra=2_5e-5.log 2>&1 &
[4]
CUDA_VISIBLE_DEVICES=3 nohup python clipprad/run.py --dataset_name AVE --batch_size 10 --lr 5e-5 --is_amp --adapter_mlp_ratio 4 > logs/AVE/20250816_xr_wall_wamp_epoch=10_ra=4_5e-5.log 2>&1 &
[8]
CUDA_VISIBLE_DEVICES=3 nohup python clipprad/run.py --dataset_name AVE --batch_size 10 --lr 5e-5 --is_amp --adapter_mlp_ratio 8 > logs/AVE/20250816_xr_wall_wamp_epoch=10_ra=8_5e-5.log 2>&1 &
[16]
CUDA_VISIBLE_DEVICES=4 nohup python clipprad/run.py --dataset_name AVE --batch_size 10 --lr 5e-5 --is_amp --adapter_mlp_ratio 16 > logs/AVE/20250816_xr_wall_wamp_epoch=10_ra=16_5e-5.log 2>&1 &
[32]
CUDA_VISIBLE_DEVICES=6 nohup python clipprad/run.py --dataset_name AVE --batch_size 10 --lr 5e-5 --is_amp --adapter_mlp_ratio 32 > logs/AVE/20250816_xr_wall_wamp_epoch=10_ra=32_5e-5.log 2>&1 &
[64]
CUDA_VISIBLE_DEVICES=7 nohup python clipprad/run.py --dataset_name AVE --batch_size 10 --lr 5e-5 --is_amp --adapter_mlp_ratio 64 > logs/AVE/20250816_xr_wall_wamp_epoch=10_ra=64_5e-5.log 2>&1 &
[128]
CUDA_VISIBLE_DEVICES=0 nohup python clipprad/run.py --dataset_name AVE --batch_size 10 --lr 5e-5 --is_amp --adapter_mlp_ratio 128 > logs/AVE/20250816_xr_wall_wamp_epoch=10_ra=128_5e-5.log 2>&1 &
[256]
CUDA_VISIBLE_DEVICES=2 nohup python clipprad/run.py --dataset_name AVE --batch_size 10 --lr 5e-5 --is_amp --adapter_mlp_ratio 256 > logs/AVE/20250816_xr_wall_wamp_epoch=10_ra=256_5e-5.log 2>&1 &







[20250814]
CUDA_VISIBLE_DEVICES=7 nohup python clipprad/run.py --dataset_name AVE --batch_size 8 --lr 5e-5 > logs/AVE/20250814_train_waudio_wmtf_5e-5.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python clipprad/run.py --dataset_name AVE --batch_size 8 --lr 5e-5 > logs/AVE/20250814_train_waudio_wmtf_focal_5e-5.log 2>&1 &


# test 

CUDA_VISIBLE_DEVICES=7 nohup python clipprad/run.py --dataset_name AVE --batch_size 8 --lr 5e-5 --continue_train_pth /home/sunchao/AVE/CLIPAdapter24/checkpoints/AVE/FinedCLIP_gpu0/2025-08-14@02-49-16 > logs/AVE/20250814_train_waudio_wmtf_wofocal_5e-5.log 2>&1 &

--continue_train_pth ./AVE/CLIPAdapter24/checkpoints/AVE/FinedCLIP_gpu0/2025-08-14@02-49-16

wausio+wofocal
CUDA_VISIBLE_DEVICES=0 nohup python clipprad/run.py --dataset_name AVE --batch_size 8 --is_amp --lr 5e-5 > logs/AVE/20250814_sc_wprob_5e-5.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python clipprad/run.py --dataset_name AVE --batch_size 10 --is_amp --lr 5e-5 > logs/AVE/20250814_sc_woprob_5e-5.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python clipprad/run.py --dataset_name AVE --batch_size 12 --is_amp --lr 5e-5 > logs/AVE/20250814_sc_womtf_5e-5.log 2>&1 &
wausio+wfocal
CUDA_VISIBLE_DEVICES=1 nohup python clipprad/run.py --dataset_name AVE --batch_size 12 --is_amp --lr 5e-5 > logs/AVE/20250814_sc_woprob_wfocal_5e-5.log 2>&1 &




[Test]
[SOTA]
--predict_ckpt ./AVE/CLIPAdapter24/checkpoints/AVE/FinedCLIP_gpu0/2025-08-14@02-49-16
CUDA_VISIBLE_DEVICES=0 nohup python clipprad/run.py --dataset_name AVE --batch_size 8 --predict_ckpt /home/sunchao/AVE/CLIPAdapter24/checkpoints/AVE/FinedCLIP_gpu0/2025-08-14@02-49-16 > logs/AVE/20250816_81.12_test1.log 2>&1 &

[2]
/home/sunchao/AVE/CLIPAdapter24/checkpoints/AVE/FinedCLIP_gpu0/2025-08-15@17-45-50
CUDA_VISIBLE_DEVICES=1 nohup python clipprad/run.py --dataset_name AVE --batch_size 24 --predict_ckpt /home/sunchao/AVE/CLIPAdapter24/checkpoints/AVE/FinedCLIP_gpu0/2025-08-15@17-45-50 > logs/AVE/20250816_r=2_test.log 2>&1 &
[4]
/home/sunchao/AVE/CLIPAdapter24/checkpoints/AVE/FinedCLIP_gpu0/2025-08-15@17-47-26
CUDA_VISIBLE_DEVICES=0 nohup python clipprad/run.py --dataset_name AVE --batch_size 24 --predict_ckpt /home/sunchao/AVE/CLIPAdapter24/checkpoints/AVE/FinedCLIP_gpu0/2025-08-15@17-47-26 > logs/AVE/20250816_r=4_test.log 2>&1 &
[8]
/home/sunchao/AVE/CLIPAdapter24/checkpoints/AVE/FinedCLIP_gpu0/2025-08-15@15-42-16
CUDA_VISIBLE_DEVICES=0 nohup python clipprad/run.py --dataset_name AVE --batch_size 24 --predict_ckpt /home/sunchao/AVE/CLIPAdapter24/checkpoints/AVE/FinedCLIP_gpu0/2025-08-15@15-42-16 > logs/AVE/20250816_r=8_test.log 2>&1 &
[16]
/home/sunchao/AVE/CLIPAdapter24/checkpoints/AVE/FinedCLIP_gpu0/2025-08-15@15-42-26
CUDA_VISIBLE_DEVICES=0 nohup python clipprad/run.py --dataset_name AVE --batch_size 24 --predict_ckpt /home/sunchao/AVE/CLIPAdapter24/checkpoints/AVE/FinedCLIP_gpu0/2025-08-15@15-42-26 > logs/AVE/20250816_r=16_test.log 2>&1 &
[32]
/home/sunchao/AVE/CLIPAdapter24/checkpoints/AVE/FinedCLIP_gpu0/2025-08-15@15-42-36
CUDA_VISIBLE_DEVICES=0 nohup python clipprad/run.py --dataset_name AVE --batch_size 24 --predict_ckpt /home/sunchao/AVE/CLIPAdapter24/checkpoints/AVE/FinedCLIP_gpu0/2025-08-15@15-42-36 > logs/AVE/20250816_r=32_test.log 2>&1 &
[64]
/home/sunchao/AVE/CLIPAdapter24/checkpoints/AVE/FinedCLIP_gpu0/2025-08-15@15-42-46
CUDA_VISIBLE_DEVICES=0 nohup python clipprad/run.py --dataset_name AVE --batch_size 24 --predict_ckpt /home/sunchao/AVE/CLIPAdapter24/checkpoints/AVE/FinedCLIP_gpu0/2025-08-15@15-42-46 > logs/AVE/20250816_r=64_test.log 2>&1 &
[128]
/home/sunchao/AVE/CLIPAdapter24/checkpoints/AVE/FinedCLIP_gpu0/2025-08-15@16-56-44
CUDA_VISIBLE_DEVICES=0 nohup python clipprad/run.py --dataset_name AVE --batch_size 24 --predict_ckpt /home/sunchao/AVE/CLIPAdapter24/checkpoints/AVE/FinedCLIP_gpu0/2025-08-15@16-56-44 > logs/AVE/20250816_r=128_test.log 2>&1 &
[256]
/home/sunchao/AVE/CLIPAdapter24/checkpoints/AVE/FinedCLIP_gpu0/2025-08-15@16-57-00
CUDA_VISIBLE_DEVICES=0 nohup python clipprad/run.py --dataset_name AVE --batch_size 24 --predict_ckpt /home/sunchao/AVE/CLIPAdapter24/checkpoints/AVE/FinedCLIP_gpu0/2025-08-15@16-57-00 > logs/AVE/20250816_r=256_test.log 2>&1 &




















