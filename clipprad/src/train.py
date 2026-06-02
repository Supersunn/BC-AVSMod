from clipprad import Summary, mkdir, exists, logger
from clipprad import get_CM, calculate_metrics, aggregate_fold_metrics
from torch.cuda.amp import autocast, GradScaler

import pandas as pd
import numpy as np
import os
import time
import torch
import torch.nn.functional as F

class BaseTrainer():
    def __init__(
        self, 
        args,
        train_loader,
        valid_loader,
        test_loader,
        model=None,
        checkpoint_log_dir=None,
        optimizer=None,
        scheduler=None,
        loss_fn=None,
        device="cpu",
        is_amp=True
    ):
        self.args = args
        logger.info(f"All params: \n {args}")
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        
        self.device = device
        self.loss_fn = loss_fn
        self.is_amp = is_amp
        if self.is_amp:
            # AMP ini
            self.scaler = GradScaler()
            logger.info("using amp")
        else:
            logger.info("don't using amp")
        
        if checkpoint_log_dir is not None:
            self.checkpoint_log_dir = checkpoint_log_dir
            self.model_root = mkdir(os.path.join(checkpoint_log_dir, "models"))
            self.log_root = mkdir(os.path.join(checkpoint_log_dir, "logs"))
            self.summary = Summary(self.log_root)
            self.model = torch.nn.DataParallel(model, device_ids=device).cuda()
            self.optimizer = optimizer
            self.scheduler = scheduler
            
            self.train_epoch = 0
            self.train_step = 0
            self.best_epoch = 0
            self.best_accuracy = 0
            self.best_loss = 1e8
            
    def train_process(self, max_epoch, wait_epoch, iteration):
        start_time = time.time()
        self.train_step = iteration
        while True:
            epoch_start_time = time.time()
            self.model.train()
            train_losses, train_predictions, train_labels = [], [], []
            clip_losses = []
            cls_losses = []
            extra_losses1, extra_losses2, extra_losses3 = [], [], []
            for idx, (batch_data, batch_label, batch_info) in enumerate(self.train_loader):
                batch_data = batch_data.cuda()
                batch_label = batch_label.cuda()
                if self.is_amp:
                    with autocast():
                        # if self.args.return_clip_loss:
                        #     outputs, clip_loss = self.model(batch_data, batch_info)
                        #     cls_loss = self.loss_fn(outputs, batch_label)
                        #     loss = clip_loss + cls_loss
                        #     clip_losses.append(clip_loss.item())
                        #     cls_losses.append(cls_loss.item())
                        # elif self.args.return_cls_loss:
                        #     outputs, (extra_out1, extra_out2, extra_out3) = self.model(batch_data, batch_info)
                        #     cls_loss = self.loss_fn(outputs, batch_label)
                        #     extra_loss1 = self.loss_fn(extra_out1, batch_label)
                        #     extra_loss2 = self.loss_fn(extra_out2, batch_label)
                        #     extra_loss3 = self.loss_fn(extra_out3, batch_label)
                        #     loss = cls_loss + extra_loss1 + extra_loss2 + extra_loss3
                        #     extra_losses1.append(extra_loss1.item())
                        #     extra_losses2.append(extra_loss2.item())
                        #     extra_losses3.append(extra_loss3.item())
                        #     cls_losses.append(cls_loss.item())
                        # else:
                        outputs = self.model(batch_data, batch_info)
                        loss = self.loss_fn(outputs, batch_label)
                        loss_value = loss.item()
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # first forward-backward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_data, batch_info)
                    loss = self.loss_fn(outputs, batch_label)
                    loss_value = loss.item()
                    loss.backward()
                    self.optimizer.first_step(zero_grad=True)
                    # self.optimizer.step()
                    
                    # second forward-backward pass
                    outputs = self.model(batch_data, batch_info)
                    loss = self.loss_fn(outputs, batch_label)
                    loss_value = loss.item()
                    loss.backward()
                    self.optimizer.second_step(zero_grad=True)
                    
                if self.train_step % 10 == 0:
                    self.summary.summary_writer_add_scalar(self.train_step, {"loss": loss_value}, tag="loss")
                    self.summary.learning_rate_summary(self.train_step, self.optimizer.param_groups, tag="step")
                    # if self.args.return_clip_loss:
                    #     logger.info(f">> Epoch: {self.train_epoch} - Step: {self.train_step} - Loss: {loss_value}, ClS part {cls_loss.item()}, CLIP part {clip_loss.item()} - Time using is: {round((time.time() - epoch_start_time) / 60, 2)} mins.")
                    # elif self.args.return_cls_loss:
                    #     logger.info(f">> Epoch: {self.train_epoch} - Step: {self.train_step} - Loss: {loss_value}, ClS part: {cls_loss.item()}, Extra part: {extra_loss1.item()}, {extra_loss2.item()}, {extra_loss3.item()} - Time using is: {round((time.time() - epoch_start_time) / 60, 2)} mins.")
                    # else:
                    logger.info(f">> Epoch: {self.train_epoch} - Step: {self.train_step} - Loss: {loss_value} - Time using is: {round((time.time() - epoch_start_time) / 60, 2)} mins.")
                self.train_step += 1
                # self.scheduler.step_update(self.train_step)
                self.scheduler.step()
                
                train_losses.append(loss_value)
                train_predictions.extend(torch.max(F.softmax(outputs, dim=-1), dim=1)[1].cpu().numpy().tolist())
                train_labels.extend(batch_label.cpu().numpy().tolist())    
            self.train_epoch += 1
            
            train_loss = np.mean(train_losses)
            train_metrics = calculate_metrics(train_predictions, train_labels)
            train_metrics = {"loss": round(train_loss, 4), **train_metrics}
            
            valid_loss, valid_metrics = self.test_process()
            valid_metrics = {"loss": round(valid_loss, 4), **valid_metrics}
            
            test_loss, test_metrics = self.test_process(is_test=True)
            test_metrics = {"loss": round(test_loss, 4), **test_metrics}
            
            self.summary.learning_rate_summary(self.train_epoch, self.optimizer.param_groups, tag="epoch")
            
            logger.info("Epoch {} - Train-Metrics: {}.".format(self.train_epoch, str(train_metrics)))
            logger.info("Epoch {} - Valid-Metrics: {}.".format(self.train_epoch, str(valid_metrics)))
            logger.info("Epoch {} - Test-Metrics: {}.".format(self.train_epoch, str(test_metrics)))
            
            self.summary.summary_writer_add_scalars(self.train_epoch, {"loss": train_loss, **train_metrics}, {"loss": valid_loss, **valid_metrics})
            
            self.save_model(valid_metrics)
            
            # erly stop
            if self.train_epoch >= max_epoch or (self.train_epoch - self.best_epoch) >= wait_epoch:
                self.summary.close()
                logger.info("Summary writer close!")
                logger.info("The epoch number is {}, and the best_epoch is {}, and the best {} is {}".format(
                    self.train_epoch, self.best_epoch, 
                    self.args.core_metrics, 
                    self.best_accuracy if self.args.optimize_direction_high else self.best_loss))
                time_consuming = time.time() - start_time
                logger.info(f"Time consume for traing: {round(time_consuming / 60, 2)} mins; {round(time_consuming / 3600, 2)} hours")
                break
        
    def test_process(self, is_test=False):
        self.model.eval()
        with torch.no_grad():
            valid_losses, valid_predictions, valid_labels = [], [], []
            
            for _, (batch_data, batch_label, batch_info) in enumerate(self.valid_loader if not is_test else self.test_loader):
                batch_data = batch_data.cuda()
                batch_label = batch_label.cuda()
                
                # if self.args.return_clip_loss:
                #     outputs, clip_loss = self.model(batch_data, batch_info)
                #     cls_loss = self.loss_fn(outputs, batch_label)
                #     loss = clip_loss + cls_loss
                # elif self.args.return_cls_loss:
                #     outputs, (extra_out1, extra_out2, extra_out3) = self.model(batch_data, batch_info)
                #     cls_loss = self.loss_fn(outputs, batch_label)
                #     extra_loss1 = self.loss_fn(extra_out1, batch_label)
                #     extra_loss2 = self.loss_fn(extra_out2, batch_label)
                #     extra_loss3 = self.loss_fn(extra_out3, batch_label)
                #     loss = cls_loss + extra_loss1 + extra_loss2 + extra_loss3
                # else:
                outputs = self.model(batch_data, batch_info)
                loss = self.loss_fn(outputs, batch_label)
                    
                valid_losses.append(loss.item())
                valid_predictions.extend(torch.max(F.softmax(outputs, dim=-1), dim=1)[1].cpu().numpy().tolist())
                valid_labels.extend(batch_label.cpu().numpy().tolist())
                
            valid_metrics = calculate_metrics(valid_predictions, valid_labels)
            valid_loss = np.mean(valid_losses)
            
        return valid_loss, valid_metrics
    
    def predict_process(self, test_pretrained_dir=None, model=None):
        models = []
        args = []
        if test_pretrained_dir is None:
            checkpoint_files = None
            checkpoint = torch.load(os.path.join(self.model_root, "best_model.pth"))
            self.model.load_state_dict(checkpoint["model"])
            models.append(self.model)
            args.append(checkpoint["args"])
        else:
            assert model is not None, "model must be initialized"
            checkpoint_files = os.listdir(test_pretrained_dir)
            for checkpoint_file in checkpoint_files:
                model_path = os.path.join(test_pretrained_dir, checkpoint_file)
                logger.info("Model Checkpoint: {}".format(model_path))
                
                checkpoint = torch.load(model_path)
                self.model = torch.nn.DataParallel(model, device_ids=self.device).cuda()
                self.model.load_state_dict(checkpoint["model"])
                models.append(self.model)
                args.append(checkpoint["args"])
        
        all_test_metrics = []
        
        for i in range(len(models)):
            model = models[i]
            model.eval()
            
            # data_loaders = {"train": self.train_loader, "valid": self.valid_loader, "test": self.test_loader}
            data_loaders = {"test": self.test_loader}
            for dataloader_key, data_loader in data_loaders.items():
                with torch.no_grad():
                    test_losses, test_predictions, test_labels = [], [], []
                    sample_ids, sample_texts, sample_captions = [], [], []
                    for _, (batch_data, batch_label, batch_info) in enumerate(data_loader):
                        batch_data = batch_data.cuda()
                        batch_label = batch_label.cuda()
                        
                        # if self.args.return_clip_loss:
                        #     outputs, clip_loss = self.model(batch_data, batch_info)
                        #     cls_loss = self.loss_fn(outputs, batch_label)
                        #     loss = clip_loss + cls_loss
                        # elif self.args.return_cls_loss:
                        #     outputs, (extra_out1, extra_out2, extra_out3) = self.model(batch_data, batch_info)
                        #     cls_loss = self.loss_fn(outputs, batch_label)
                        #     extra_loss1 = self.loss_fn(extra_out1, batch_label)
                        #     extra_loss2 = self.loss_fn(extra_out2, batch_label)
                        #     extra_loss3 = self.loss_fn(extra_out3, batch_label)
                        #     loss = cls_loss + extra_loss1 + extra_loss2 + extra_loss3
                        # else:
                        outputs = self.model(batch_data, batch_info)
                        loss = self.loss_fn(outputs, batch_label)
                            
                        test_losses.append(loss.item())
                        test_predictions.extend(torch.max(F.softmax(outputs, dim=-1), dim=1)[1].cpu().numpy().tolist())
                        test_labels.extend(batch_label.cpu().numpy().tolist())
                        sample_ids.extend(batch_info["id"].cpu().numpy().tolist())
                        sample_texts.extend(batch_info["text"])
                        sample_captions.extend(batch_info["frame_caption"])
                        
                    df = pd.DataFrame({
                        'Id': sample_ids,
                        'Text': sample_texts,
                        'Caption': sample_captions,
                        'Prediction': test_predictions,
                        "Label": test_labels,
                    })
                    
                    df.to_csv(f'{dataloader_key}_output.csv', index=False)
                    
                    test_metrics = calculate_metrics(test_predictions, test_labels)
                    test_cm = get_CM(test_labels, test_predictions)
                    
                    logger.info("Loss for {}: {}".format(dataloader_key, round(np.mean(test_losses), 4)))
                    logger.info("Metrics for {}: {}".format(dataloader_key, str(test_metrics)))
                    logger.info("CM for {}: \n{}".format(dataloader_key, test_cm))
                    if dataloader_key == "test":
                        all_test_metrics.append(test_metrics)
                        
                    
                        
        test_metrics = aggregate_fold_metrics(all_test_metrics)
        return test_metrics, args[0]
        
    def save_model(self, metrics):
        if self.args.optimize_direction_high:
            
            if metrics[self.args.core_metrics] >= (self.best_accuracy + 1e-6):
                self.best_epoch = self.train_epoch
                self.best_accuracy = metrics[self.args.core_metrics]
                model_save_path = os.path.join(self.model_root, "best_model.pth")
                torch.save(
                    {
                        "args": self.args,
                        "epoch": self.train_epoch,
                        "metrics": metrics,
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict() if exists(self.scheduler) else None,
                    },
                    model_save_path,
                )
        else:
            if metrics[self.args.core_metrics] <= (self.best_loss - 1e-6):
                
                self.best_epoch = self.train_epoch
                self.best_loss = metrics[self.args.core_metrics]
                model_save_path = os.path.join(self.model_root, "best_model.pth")
                torch.save(
                    {
                        "args": self.args,
                        "epoch": self.train_epoch,
                        "step": self.train_step,
                        "metrics": metrics,
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict() if exists(self.scheduler) else None,
                    },
                    model_save_path,
                )
                