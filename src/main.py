import argparse
from train import trainModel
from test import testModel

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t',
                        '--train',
                        action='store_true',
                        help='train the model',
                        dest='train')

    return parser.parse_args()

if __name__ == '__main__':
    args = parseArguments()

    if args.train:
        trainModel()
    else:
        testModel()
