from clipprad.src.datasets.mmsd2 import AVEDataset
from clipprad.src.datasets.mmsd import MMSDDataset

def get_instance(name, parameters_dict):
    model = {
        'MMSD': MMSDDataset,
        'AVE': AVEDataset,
    }[name]
    return model(**parameters_dict)
