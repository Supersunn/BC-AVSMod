import json
import os
import time
import traceback

import torch

from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup
from timm.scheduler.cosine_lr import CosineLRScheduler
from clipprad.src.models.loss.loss import loss_fn_dict
from clipprad.configs.config import Config
from clipprad.src.get_dataloader import get_base_dataloader
from clipprad.src.utils.get_basictools import (check_and_create_dir, set_seed, write_json_result_to_csv, write_single_json_to_whole)
from clipprad import aggregate_fold_metrics
from clipprad.src.models.optimizer.sam import SAM
import clipprad.src.models as all_models
from clipprad.src.train import BaseTrainer
from clipprad.configs.mmsd_input_args import parse_args
input_args = parse_args()

from clipprad import logger

os.environ["CUDA_VISIBLE_DEVICES"] = input_args.gpu_id
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"                                       # This is crucial for reproducibility
logger.info(f"Nvidia number: {os.environ['CUDA_VISIBLE_DEVICES']}")

torch.set_float32_matmul_precision("medium")

results_number = 0                                      
result_list_path = ''                                   

is_dataloader_finished = False
train_dataloaders, valid_dataloaders, test_dataloaders = [], [], []

def objective():
    global_args = Config(input_args).get_params()
    global_args.checkpoint_log_path = os.path.join(global_args.work_dir, global_args.checkpoint_log_path)
    global_args.res_save_path = os.path.join(global_args.work_dir, global_args.res_save_path)
    
    time_dir = time.strftime("%Y-%m-%d@%H-%M-%S")
    
    if global_args.predict_ckpt is None:
        global_args.checkpoint_log_path = check_and_create_dir(global_args, global_args.checkpoint_log_path, time_dir)
        global_args.res_save_path = check_and_create_dir(global_args, global_args.res_save_path)
    global result_list_path
    result_list_path = global_args.res_save_path
    
    set_seed(global_args.seed)
    
    if global_args.predict_ckpt is None:
        with open(os.path.join(global_args.checkpoint_log_path, "args_file.json"), "w+") as file:
            json.dump(global_args, fp=file, indent=4)
            
    test_metrics, train_args = do_train(global_args)
    
    global results_number     
    results_number += 1
    
    single_trail_results = {
        "id": results_number,
        "metrics": test_metrics,
        "params": train_args,
    }
    if global_args.predict_ckpt is None:
        trail_result_path = os.path.join(global_args.res_save_path, f"{input_args.extra_msg}_{results_number}.json")
        with open(trail_result_path, "w+") as file:
            json.dump(single_trail_results, fp=file, indent=4)
        logger.info("Experiment results are save to {}.\n".format(trail_result_path))  
    
    return test_metrics[global_args.core_metrics]

def do_train(args):
    device = [int(i) for i in args.gpu_id.split(",")]
    logger.info(f"Device id is: {device}")
    global is_dataloader_finished
    if not is_dataloader_finished:
        global train_dataloaders, test_dataloaders
        train_dataloader, valid_dataloader, test_dataloader = get_base_dataloader(args)
        train_dataloaders.append(train_dataloader)
        valid_dataloaders.append(valid_dataloader)
        test_dataloaders.append(test_dataloader)
        is_dataloader_finished = True
    
    all_fold_metrics = []
    
    for ii in range(len(train_dataloaders)):
        ckpt_log_subdir = os.path.join(args.checkpoint_log_path, str(ii))
        if args.predict_ckpt is None:
            if not os.path.exists(ckpt_log_subdir):
                os.makedirs(ckpt_log_subdir)
        
        model = all_models.get_instance(
            name=args.net_type, 
            parameters_dict={
                **args.model_config, 
            },
        )
        
        filtered_prompt_params = []
        for name, param in model.named_parameters():
            if 'ctx_params' in name and param.requires_grad:
                filtered_prompt_params.append(param)
        
        filtered_adapter_params = []
        for name, param in model.named_parameters():
            if 'adapter' in name and param.requires_grad:
                filtered_adapter_params.append(param)
                
        filtered_other_params = []
        for name, param in model.named_parameters():
            if ('adapter' not in name and 'ctx_params' not in name) and param.requires_grad:
                filtered_other_params.append(param)
                
        logger.info(f"filtered_prompt_params: {sum(p.numel() for p in filtered_prompt_params if p.requires_grad)}")
        logger.info(f"filtered_adapter_params: {sum(p.numel() for p in filtered_adapter_params if p.requires_grad)}")
        logger.info(f"filtered_other_params: {sum(p.numel() for p in filtered_other_params if p.requires_grad)}")
        logger.info(f" Num of all params: {sum(p.numel() for p in model.parameters())}")
        logger.info(f" Num of learnable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        # define optimizer
        if args.with_amp:
            optimizer = torch.optim.AdamW(
                params=[
                    # {'params': filtered_prompt_params, 'lr': args.lr}, 
                    {'params': filtered_adapter_params, 'lr': args.lr},
                    {'params': filtered_other_params, 'lr': args.lr}
                ], # old lr for all
                lr=args.lr,
                # momentum=0.9,
                # weight_decay=1e-4
            )
        else:  
        #     optimizer = torch.optim.AdamW(
        #         params=[
        #             {'params': filtered_prompt_params, 'lr': args.lr}, 
        #             {'params': filtered_adapter_params, 'lr': args.lr},
        #             {'params': filtered_other_params, 'lr': args.lr}
        #         ], # old lr for all
        #         # lr=args.lr,
        #         # momentum=0.9,
        #         # weight_decay=1e-4
        #     )
            optimizer = SAM(
                params=[
                    # {'params': filtered_prompt_params, 'lr': args.lr}, 
                    {'params': filtered_adapter_params, 'lr': args.lr},
                    {'params': filtered_other_params, 'lr': args.lr}
                ],
                base_optimizer=torch.optim.AdamW,
                lr=args.lr,
                # momentum=0.9,
                # weight_decay=1e-4,
                rho=0.05, 
                adaptive=False
            )
        
        # scheduler = CosineLRScheduler(
        #     optimizer,
        #     t_initial=int(args.max_epoch * len(train_dataloaders[ii])),
        #     lr_min=5e-6,
        #     warmup_lr_init=5e-7,
        #     warmup_t=int(5 * len(train_dataloaders[ii])),
        #     cycle_limit=1,
        #     t_in_epochs=False,
        # )
        
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.2 * args.max_epoch * len(train_dataloaders[ii])),
            num_training_steps=int(args.max_epoch* len(train_dataloaders[ii])),
            min_lr_rate=0.01,
        )
        
        if args.predict_ckpt is None:
            logger.info('*' * 10 + f' Folder {ii + 1} begin to train ' + '*' * 10)
            train_process = BaseTrainer(
                args=args,
                train_loader=train_dataloaders[ii],
                valid_loader=valid_dataloaders[ii],
                test_loader=test_dataloaders[ii],
                model=model,
                checkpoint_log_dir=ckpt_log_subdir,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                loss_fn=loss_fn_dict[args.loss_fn_name],
                is_amp=args.with_amp
            )
            train_process.train_process(
                max_epoch=args.max_epoch,
                wait_epoch=args.early_stop,
                iteration=0,
            )
            test_metrics, train_args = train_process.predict_process()
            all_fold_metrics.append(test_metrics)
        else:
            logger.info('*' * 10 + f' Folder {ii + 1} begin to predict ' + '*' * 10)
            model_ckpt_pth = os.path.join(args.predict_ckpt, f"{ii}", "models")
            
            train_process = BaseTrainer(
                args=args,
                train_loader=train_dataloaders[ii],
                valid_loader=valid_dataloaders[ii],
                test_loader=test_dataloaders[ii],
                device=device,
                loss_fn=loss_fn_dict[args.loss_fn_name],
                is_amp=args.with_amp
            )
            
            test_metrics, train_args = train_process.predict_process(model_ckpt_pth, model)
            all_fold_metrics.append(test_metrics)  
    test_metrics = aggregate_fold_metrics(all_fold_metrics)
    
    logger.info('=' * 10 + "Cross validation summary start" + '=' * 10)
    for ii in range(len(all_fold_metrics)):
        logger.info(f"Folder {ii + 1}: {all_fold_metrics[ii]}.")
    logger.info(f"Mean: {test_metrics}")
    logger.info('=' * 10 + "Cross validation summary end" + '=' * 10)
    
    return test_metrics, train_args

if __name__ == "__main__":
    try:
        objective()
    except Exception as e:
        logger.info("Study trails are not finished, some errors occurred!")
        traceback.print_exc()
    finally:
        if input_args.predict_ckpt is None:
            write_json_result_to_csv(
                json_dir=result_list_path,
                csv_output=os.path.join(result_list_path, f"results_{input_args.net_type}_{input_args.extra_msg}.csv"),
                prefix=f"{input_args.extra_msg}"
            )
            write_single_json_to_whole(
                json_dir=result_list_path,
                json_output=os.path.join(result_list_path, f"results_{input_args.net_type}_{input_args.extra_msg}.json"),
                prefix=f"{input_args.extra_msg}"
            )
        