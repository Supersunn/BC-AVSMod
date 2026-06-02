source ~/.zshrc
conda activate msd
cd ./Documents/competition/BC-AVSMod
export PYTHONPATH="./Documents/competition/BC-AVSMod:$PYTHONPATH"

# bses=(
#     8
#     16
#     24
#     32
#     48
#     56
#     64
# )
# for bs in "${bses[@]}"; do
#     CUDA_VISIBLE_DEVICES=2 python clipprad/run.py --dataset_name MMSD --net_type FinedCLIP --num_workers 4 --batch_size $bs --gpu_id 0 --extra_msg bs="$bs"
# done

# lrs=(
#     1e-5
#     3e-5
#     5e-5
#     1e-4
#     3e-4
#     5e-4
#     1e-3
# )
# for lr in "${lrs[@]}"; do
#     CUDA_VISIBLE_DEVICES=0 python clipprad/run.py --dataset_name MMSD --net_type FinedCLIP --num_workers 4 --batch_size 64 --lr $lr --gpu_id 0 --extra_msg lr="$lr"
# done

# adapter_mlp_ratios=(
#     32
#     16
#     8
#     4
# )
# for adapter_mlp_ratio in "${adapter_mlp_ratios[@]}"; do
#     CUDA_VISIBLE_DEVICES=1 python clipprad/run.py --dataset_name MMSD --net_type FinedCLIP --num_workers 4 --batch_size 64 --adapter_mlp_ratio $adapter_mlp_ratio --gpu_id 0 --extra_msg adapter_mlp_ratio="$adapter_mlp_ratio"
# done

# prompt_n_ctxs=(
#     4
#     8
#     10
#     16
# )
# for prompt_n_ctx in "${prompt_n_ctxs[@]}"; do
#     CUDA_VISIBLE_DEVICES=5 python clipprad/run.py --dataset_name MMSD --net_type FinedCLIP --num_workers 4 --batch_size 64 --prompt_n_ctx $prompt_n_ctx --gpu_id 0 --extra_msg prompt_n_ctx="$prompt_n_ctx"
# done

att_adapter_poses=(
    series
)
for att_adapter_pos in "${att_adapter_poses[@]}"; do
    CUDA_VISIBLE_DEVICES=4 python clipprad/run.py --dataset_name MMSD --net_type FinedCLIP --num_workers 4 --batch_size 64 --att_adapter_pos $att_adapter_pos --gpu_id 0 --extra_msg att_adapter_pos="$att_adapter_pos"
done