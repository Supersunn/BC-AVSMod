import importlib
import os
import pandas as pd
import numpy as np
import glob
import json
import random
import shutil
import torch
import yaml


class Storage(dict):
    """
    A Storage object is like a dictionary except `obj.foo` can be used indication to `obj['foo']`
    ref: https://blog.csdn.net/a200822146085/article/details/88430450
    """
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)
        
    def __setattr__(self, key, value):
        self[key] = value
        
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)
        
    def __str__(self):
        return  dict.__repr__(self)
    
def set_seed(seed):
    """
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
def get_class_from_name(name):
    args = name.split('.')
    package_name = ''
    preprocess_class_name = args[-1]
    for i in range(len(args) - 1):
        package_name += args[i] + '.'
    package_name = package_name[0: -1]
    
    preprocess_module = importlib.import_module(package_name)
    preprocess_class = getattr(preprocess_module, preprocess_class_name)
    return preprocess_class

def check_and_create_dir(args, path, time_dir=None, sub_dir=None):

    if not os.path.exists(path):
        os.makedirs(path)
        
    dataset_dir = os.path.join(path, args.dataset_name)
    if not os.path.join(path, dataset_dir):
        os.makedirs(dataset_dir)
        
    net_dir = os.path.join(dataset_dir, f"{args.net_type}_gpu{args.gpu_id}")
    
    if args.dataset_name == "iemocap":
        net_dir = os.path.join(dataset_dir, f"{args.net_type}_Session{args.debug_test_id}_gpu{args.gpu_id}")
    else:
        net_dir = os.path.join(dataset_dir, f"{args.net_type}_gpu{args.gpu_id}")
    
    if not os.path.exists(net_dir):
        os.makedirs(net_dir)
        
    if time_dir is None:
        return net_dir
    else:
        path = os.path.join(net_dir, time_dir)
        if not os.path.exists(path):
            os.makedirs(path)
        
        if sub_dir is None:
            return path
        else:
            sub_path = os.path.join(path, sub_dir)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)
            return sub_path
        
def deep_select_dict(dict1, prefix='', delimiter='/'):
    results = {}
    for key, value in dict1.items():
        if isinstance(value, dict):
            results = {**results, **deep_select_dict(value, key)}
        elif isinstance(value, torch.Tensor):
            results[prefix + delimiter + key] = value.cpu().tolist()
        else:
            results[prefix + delimiter + key] = value
    return results

def write_json_result_to_csv(json_dir, csv_output, prefix):

    jsonfile_list = glob.glob(f"{json_dir}/{prefix}*.json")
    head, results = [], []
    for jsonfile in jsonfile_list:
        with open(jsonfile, 'r') as file:
            data = json.load(file)
            if type(data) == list:
                pass
            elif type(data) == dict:
                data = [data]
            for i in range(len(data)):
                if len(head) == 0:
                    head.append('id')
                    head.extend(list(deep_select_dict(data[i]['metrics']).keys()))
                    head.extend(list(deep_select_dict(data[i]['params']).keys()))
            
                result = []
                result.append(data[i]['id']),
                result.extend(list(deep_select_dict(data[i]['metrics']).values()))
                result.extend(list(deep_select_dict(data[i]['params']).values()))
                results.append(result)
            

    df = pd.DataFrame(columns=head, data=results)
    df.to_csv(csv_output, index=False)
    
def write_single_json_to_whole(json_dir, json_output, prefix):

    jsonfile_list = glob.glob(f"{json_dir}/{prefix}_*.json")
    results = []
    for jsonfile in jsonfile_list:
        with open(jsonfile, 'r') as file:
            data = json.load(file)
            results.append(data)
            
    with open(json_output, "w+") as file:
        json.dump(results, fp=file, indent=4)
    print("Experiment results are unified to {}.\n".format(json_output))
    
def save_dict_to_yaml(dict_value: dict, save_path: str):
    """dict saved as yaml"""
    with open(save_path, 'w') as file:
        file.write(yaml.dump(dict_value, allow_unicode=True))

def read_yaml_to_dict(yaml_path: str, ):
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value

def exists(x):
    return x is not None

def mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    return dir_path
