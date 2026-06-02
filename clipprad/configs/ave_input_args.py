import argparse
from clipprad import WORK_PATH, DATA_PATH, PARALLEL_ADAPTER, SERIES_ADAPTER


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name",type=str, default="AVE",help="support MMSD, AVE")
    parser.add_argument("--net_type", type=str, default="FinedCLIP", help="support FinedCLIP")
    parser.add_argument("--num_workers", type=int, default=4, help="num workers of loading data")

    parser.add_argument("--work_dir", type=str, default=WORK_PATH, help="path of working directory")
    parser.add_argument("--dataset_prefix", type=str, default=DATA_PATH, help="path to dataset prefix")
    parser.add_argument("--checkpoint_log_path", type=str, default="checkpoints", help="path to save model and correspond tensorboard log.")
    parser.add_argument("--res_save_path", type=str, default="results", help="path to save results.")
    parser.add_argument("--predict_ckpt", type=str, default=None, help="default mode for train or predict")
    
    parser.add_argument("--gpu_id", default="0", help="⭐️ gpu ID")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--with_amp", action='store_true', help="⭐️ AMP")
    parser.add_argument("--lr", default=1e-4, type=float)
    
    parser.add_argument("--att_adapter_pos", default=PARALLEL_ADAPTER, type=str)
    parser.add_argument("--mlp_adapter_pos", default=PARALLEL_ADAPTER, type=str)
    parser.add_argument("--adapter_mlp_ratio", default=8, type=int)
    parser.add_argument("--prompt_n_ctx", default=8, type=int)
    parser.add_argument("--projection_dim", default=512, type=int)
    parser.add_argument("--fusion_drp", default=0.5, type=float)
    parser.add_argument("--fusion_expand", default=4, type=int)
    
    parser.add_argument("--extra_msg", default="", help="⭐️ extra messages")
    return parser.parse_args()
