from clipprad.src.utils.get_basictools import Storage
import os


class Config():
    def __init__(self, args) -> None:
        self.dataset_name = args.dataset_name
        self.net_type = args.net_type
        self.work_dir = args.work_dir
        self.dataset_dir = args.dataset_prefix
        self.lr = args.lr
        self.att_adapter_pos = args.att_adapter_pos
        self.mlp_adapter_pos = args.mlp_adapter_pos
        self.adapter_mlp_ratio = args.adapter_mlp_ratio
        self.prompt_n_ctx = args.prompt_n_ctx
        self.projection_dim = args.projection_dim
        self.fusion_drp = args.fusion_drp
        self.fusion_expand = args.fusion_expand
        
        assert self.dataset_name in ["MMSD", "AVE"], "support MMSD, AVE, dataset."
        try:
            self.global_params = vars(args)
        except TypeError:
            self.global_params = args
            
    def net(self):
        net_cfg = {
            "FinedCLIP": {
                # "text_prompt": [
                #     "same meaning, objective description, understatement, free of criticism, and neutral tone",
                #     "opposite or contradictory meanings, exaggeration, subtle criticism, and humorous or sarcastic tone",
                # ],
                "ctx_init": None,
                # "text_prompt": [
                #     "a straightforward situation with no sarcasm",
                #     "a situation rich in sarcasm and irony",
                # ],
                "text_prompt": [
                    "no AVE",
                    "AVE",
                ],
                # "ctx_init": "This multi-modal input suggests that this image and additional data together depict",
                "prompt_n_ctx": self.prompt_n_ctx,
                "att_adapter_pos": self.att_adapter_pos,
                "mlp_adapter_pos": self.mlp_adapter_pos,
                "adapter_mlp_ratio": self.adapter_mlp_ratio,
                "projection_dim": self.projection_dim,
                "fusion_drp": self.fusion_drp,
                "fusion_expand": self.fusion_expand,
                "pretrained_pth": "pretrained_llms/openai/clip-vit-large-patch14-336",
                "text_pretrained_pth": "pretrained_llms/BeichenZhang/longclip-L.pt"
            },
        }
        return net_cfg
    
    def dataset(self):
        datset_cfg = {
            "MMSD": {
                "dataset_class": "MMSD",
                "dataset_dir": {
                    "image": os.path.join(self.work_dir, self.dataset_dir, "images"),
                    "caption": "./Documents/competition/MSDCLIPAdapter/LLaMA/AVE_caption_fined_clean.json",
                    # "caption": os.path.join(self.work_dir, self.dataset_dir, "captions/AVE_caption_clean.json"),
                    "label": {
                        "train": os.path.join(self.work_dir, self.dataset_dir, "version1/train.txt"),
                        "valid": os.path.join(self.work_dir, self.dataset_dir, "version1/valid2.txt"),
                        "test": os.path.join(self.work_dir, self.dataset_dir, "version1/test2.txt")
                    }
                },
                "class_number": 2,
                "emotion_to_label": {'sarcastic': 1, 'not sarcastic': 0},
                "clip_pretrained_path": "pretrained_llms/openai/clip-vit-large-patch14-336"
            },
            "AVE": {
                "dataset_class": "AVE",
                "dataset_dir": {
                    "image": os.path.join(self.work_dir, self.dataset_dir, "images"),
                    # "caption": os.path.join(self.work_dir, self.dataset_dir, "captions/AVE_data.json"),
                    # "caption": os.path.join(self.work_dir, self.dataset_dir, "captions/AVE_caption_clean.json"),
                    # "caption": "/home/zhuchuanbo/Documents/competition/MSDCLIPAdapter/LLaMA/AVE_llava_caption_clean.json",
                    # "caption": "/home/zhuchuanbo/Documents/competition/MSDCLIPAdapter/data/AVE_caption_fined.json",
                    "caption": "./Documents/competition/MSDCLIPAdapter/LLaMA/AVE_caption_fined_clean.json",
                    "label": {
                        "train": os.path.join(self.work_dir, self.dataset_dir, "version2/train.json"),
                        "valid": os.path.join(self.work_dir, self.dataset_dir, "version2/valid.json"),
                        "test": os.path.join(self.work_dir, self.dataset_dir, "version2/test.json")
                    },
                },
                "class_number": 2,
                "emotion_to_label": {'sarcastic': 1, 'not sarcastic': 0},
                "clip_pretrained_path": "pretrained_llms/openai/clip-vit-large-patch14-336"
            }
        }
        return datset_cfg[self.dataset_name]
    
    def optimize(self):
        opt_cfg = {
            "MMSD": {
                "loss_fn_name": "sce",
                "core_metrics": "Accuracy",
                "optimize_direction_high": True,
                "train_type": "Base",
                "early_stop": 10,  # default is 50
                "max_epoch": 10,
            },
            "AVE": {
                "loss_fn_name": "sce",
                "core_metrics": "Accuracy",
                "optimize_direction_high": True,
                "train_type": "Base",
                "early_stop": 5,
                "max_epoch": 5,
            },
        }
        return opt_cfg[self.dataset_name]
    
    def optuna(self):
        return {
            "lr": self.lr,
        }
        
    def get_params(self):
        e = self.global_params
        a = self.net()[self.net_type]
        b = self.dataset()
        c = self.optimize()
        d = self.optuna()
        return Storage(
            {
                **e,
                **{"model_config": a},
                **c,
                **b,
                **d,
            }
        )
