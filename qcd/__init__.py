""" Global QCD exports """
from typing import Optional
import pickle


def load_object_from_file(name: str, path: Optional[str] = ""):
    """ load object from a file """
    with open(f'./{path}{name}.pkl', 'rb') as file:
        return pickle.load(file)


def save_object_to_disk(obj, name: str, path: Optional[str] = "") -> None:
    """ save result to a file """
    with open(f'./{path}{name}.pkl', 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)
