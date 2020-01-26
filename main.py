import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-models', '-m')
parser.add_argument('-validation', '-v')
parser.add_argument('-test-size', '-s')
parser.add_argument('-n-splits', '-n')
parser.add_argument('-file', '-f')

args = parser.parse_args()
models = args.models
validation = args.validation
file = args.file
test_size = args.test_size
n_splits = int(args.n_splits)

# python main.py -models cart -validation split -file german.txt
# python main.py -m cart -v split -f german.txt -s 0.3

from helpers.models import run_cart, run_cart_kfold

if models == 'cart':
    if validation == 'split':
        run_cart(f'data/{file}', test_size)
    elif validation == 'cross':
        run_cart_kfold(f'data/{file}', n_splits)