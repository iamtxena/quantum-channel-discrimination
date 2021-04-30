""" Global QCD exports """
from qcd.optimizationresults.globaloptimizationresults.global_aux import get_max_result
from qcd.typings import CloneSetup
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


def merge_results_to_one_file(clone_setup: CloneSetup) -> None:
    """ Load the split results files generated with different clones and
        merge into a single results object that is also saved to disk
        using the same base name """
    if clone_setup is None or not ('total_clones' in clone_setup) or not ('file_name' in clone_setup):
        raise ValueError('CloneSetup required with total_clones and file_name')

    loaded_results = []
    for idx in range(clone_setup['total_clones']):
        loaded_results.append(load_object_from_file(f"{clone_setup['file_name']}_{idx}", clone_setup['path']))

    merged_results = get_max_result(loaded_results)

    save_object_to_disk(merged_results, clone_setup['file_name'], clone_setup['path'])
