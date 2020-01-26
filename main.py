import argparse
parser = argparse.ArgumentParser()

# command line arguments
parser.add_argument('-models', '-m') # options: cart, cart-abc
parser.add_argument('-validation', '-v') # options: split, cross
parser.add_argument('-file', '-f') # filename in data directory

parser.add_argument('-test-size', '-s') # float 0.0 - 1.0 for -v split
parser.add_argument('-n-splits', '-n') # unsigned integer for -v cross
parser.add_argument('-modification-rate', '-mr') # float 0.0 - 1.0 for -m cart-abc
parser.add_argument('-cycle', '-c') # unsigned integer for -m cart-abc


args = parser.parse_args()
models = args.models
validation = args.validation
file = args.file
test_size = float(args.test_size)
n_splits = int(0 if args.n_splits == None else args.n_splits)
modification_rate = float(0.0 if args.modification_rate == None else args.modification_rate)
cycle = int(0 if args.cycle == None else args.cycle)

# python main.py -models cart -validation split -file german.txt
# python main.py -m cart -v split -f german.txt -s 0.3

"""
NOTES !!

True Positive Rate (TPR) = Sensitivity = Recall
True Negative Rate (TNR) = 1 - FPR = Specificity
"""

from helpers.models import run_cart, run_cart_kfold, run_cart_abc

if models == 'cart':
    if validation == 'split':
        run_cart(f'data/{file}', test_size)
    elif validation == 'cross':
        run_cart_kfold(f'data/{file}', n_splits)

elif models == 'cart-abc':
    if validation == 'split':
        run_cart_abc(f'data/{file}', test_size)