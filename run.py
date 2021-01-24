from userlibs import *

import warnings
import argparse


""" Arguments """
def parse_args():

    parser = argparse.ArgumentParser(description='Multilayer perceptron example')
    parser.add_argument('--training-data', type=str, help='training data set in csv')
    parser.add_argument('--conda-env', type=str, default=None,
                        help='the path to a conda environment yaml file (default: None)')
    return parser.parse_args()


def main():

    warnings.filterwarnings("ignore")
    args = parse_args()

    """ data """
    data = read_raw_data(args.training_data) # mandatory

    """ model """
    mlflow_model(data)


if __name__ == "__main__":
	main()
    
