# eidas_demo.py
# flake8: noqa
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from qcd.dampingchannels import OneShotEntangledDampingChannel
from qcd import save_object_to_disk


def main():
    print("Testing Optimization")
    optimization_setup = {'optimizer_algorithms': ['DIRECT_L'],
                          'optimizer_iterations': [1000],
                          'eta_partitions': 3,  # number of partitions for the eta ranging from 0 to pi/2
                          'number_channels_to_discriminate': 3,
                          'number_third_channels': 20,
                          'plays': 10000}

    results = OneShotEntangledDampingChannel.find_optimal_configurations(optimization_setup=optimization_setup)
    filename = '20210417_C2_A2_1000_10000_3_6_20'
    print(f'saving results to {filename}')
    save_object_to_disk(results, name=filename, path="demo/data/")


if __name__ == "__main__":
    main()
