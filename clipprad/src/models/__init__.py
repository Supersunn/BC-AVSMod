from clipprad.src.models.model import FinedCLIP

def get_instance(name, parameters_dict):
    model = {
        "FinedCLIP": FinedCLIP,
    }[name]
    return model(**parameters_dict)

