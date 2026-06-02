import numpy as np
import torch
import torchvision
import os
import cv2
import random
import json

from torch.utils.data.dataset import Dataset
from clipprad import logger
from transformers import CLIPProcessor
from PIL import Image

class AVEDataset(Dataset):
    def __init__(
        self, 
        dataset_dir, 
        class_number=2, 
        mode="train", 
        emotion_to_label=None,
        clip_pretrained_path=None
    ):
        super(AVEDataset, self).__init__()
        assert mode in ["train", "valid", "test"], "only support train, valid and test"
        
        self.mode = mode
        self.number_classes = class_number
        self.image_size = 336 if "large" in clip_pretrained_path else 224
        
        self.image_dir = dataset_dir['image']
        self.caption_path = dataset_dir['caption']
        self.label_path = dataset_dir['label']
        
        self.clip_pretrained_path = clip_pretrained_path
        
        if self.clip_pretrained_path is not None:
            self.processor = CLIPProcessor.from_pretrained(self.clip_pretrained_path)
        else:
            self.processor = self.get_transform()
            
        self.basic_aug = mode == "train"
        self.aug_func = [flip_image, add_gaussian_noise]
        
        self.emotion_class2id = emotion_to_label
        self.emotion_id2class = dict(zip(self.emotion_class2id.values(), self.emotion_class2id.keys()))
        self.emotion_label_count = dict(zip(self.emotion_class2id.keys(), np.zeros(self.number_classes, dtype=np.int64())))
        self.length = 0
        
        self.data_dict = {}
        
        self.__create_sample_label_dict()
    
    def get_transform(self):
        transform = None
        if self.mode == "train":
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.Resize((self.image_size, self.image_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                torchvision.transforms.RandomErasing(scale=(0.02, 0.1)),
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((self.image_size, self.image_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            )
        return transform
    
    def __create_sample_label_dict(self):
        f1= open(self.label_path[self.mode], 'r', encoding='utf-8')
        datas = json.load(f1)
        caption_dict = json.load(open(self.caption_path, 'r', encoding='utf-8'))
        for data in datas:
            self.data_dict[self.length] = {
                "string_id": data['image_id'],
                "frame_path": os.path.join(self.image_dir, f"{str(data['image_id'])}.jpg"),
                "frame_caption": caption_dict[str(data['image_id'])],
                "text": data["text"],
                "emotion_label": data['label'],
                "emotion_name": self.emotion_id2class[data['label']]
            }
            self.emotion_label_count[self.emotion_id2class[data['label']]] += 1
            self.length += 1
            
        logger.info(f"{'*' * 100}")
        logger.info(f"Dataset mode: {self.mode}; Sample number: {self.length}")
        logger.info(f"Emotion label statistics: {self.emotion_label_count}")
        
    def __getitem__(self, index):
        data = self.data_dict[index]
        img = Image.open(data["frame_path"])
        
        # if self.basic_aug:
        #     if random.uniform(0, 1) > 0.5:
        #         index = random.randint(0, 1)
        #         img = self.aug_func[index](img)
        
        if self.clip_pretrained_path is not None:
            img = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze()
        else:
            img = self.processor(img.copy())
        
        return (
            img,
            torch.tensor(data["emotion_label"]).long(),
            {
                "id": data["string_id"],
                "text": data["text"],
                "frame_caption": data["frame_caption"],
                "cls_label": data["emotion_label"],
                "cls_name": data["emotion_name"],
            }
        )
        
    def __len__(self):
        return self.length
    
def add_gaussian_noise(image_array, mean=0.0, var=30):
    std = var**0.5
    noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped

def flip_image(image_array):
    return cv2.flip(image_array, 1)
    