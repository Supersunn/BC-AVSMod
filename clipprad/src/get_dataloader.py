import torch
import clipprad.src.datasets as dataloaders

from clipprad import logger


def get_base_dataloader(args, data_split_set=1):
    data_loader_params = {}
    data_loader_params = {
        "dataset_dir": args.dataset_dir, 
        "class_number": args.class_number, 
        "emotion_to_label": args.emotion_to_label,
        "clip_pretrained_path": args.clip_pretrained_path
    }
    logger.info(f"********** {args.dataset_name} dataset message **********")
    data_loader_params["mode"] = "train"
    train_dataloader = torch.utils.data.DataLoader(
        dataloaders.get_instance(args.dataset_class, data_loader_params),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    data_loader_params["mode"] = "valid"
    valid_dataloader = torch.utils.data.DataLoader(
        dataloaders.get_instance(args.dataset_class, data_loader_params),
        batch_size=args.batch_size * 4,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    data_loader_params["mode"] = "test"
    test_dataloader = torch.utils.data.DataLoader(
        dataloaders.get_instance(args.dataset_class, data_loader_params),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    return train_dataloader, valid_dataloader, test_dataloader        
    