import warnings
import logging

SERIES_ADAPTER = "series"
PARALLEL_ADAPTER = "parallel"
warnings.filterwarnings("ignore")

from clipprad.src.utils.get_summary import Summary
from clipprad.src.utils.get_basictools import mkdir, exists
from clipprad.src.utils.get_metrics import get_CM, calculate_metrics, aggregate_fold_metrics


def singleton(cls):
    instances = {}
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return wrapper

@singleton
class SingleLogger():
    def get_static_logger(self, name='clipprad'):
        logging.basicConfig(
            level=logging.DEBUG, 
            datefmt='%Y/%m/%d %H:%M:%S', 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(name)
        return logger
    
    def get_logger_id(self):
        return id(self)
    
logger_thing = SingleLogger()
logger = logger_thing.get_static_logger()
logger.info(f"Logger instance id: {logger_thing.get_logger_id()}")

image_logger = logging.getLogger('PIL.PngImagePlugin')
image_logger.setLevel(logging.ERROR)

class ModuleFilter(logging.Filter):
    def filter(self, record):
        return not (record.name == 'PIL.PngImagePlugin' and record.levelno == logging.DEBUG)


image_logger.addFilter(ModuleFilter())
