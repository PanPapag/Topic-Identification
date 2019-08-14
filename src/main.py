import argparse

from app.app import *

def make_args_parser():
    # create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Applied machine learning in automated text categorization')
    # fill parser with information about program arguments
    parser.add_argument('--datasets', default='datasets',
                         help='Define the relative path of the dataset directory')
    parser.add_argument('--outputs', default='outputs',
                         help='Define the relative path of the output directory')
    parser.add_argument('--duplicates', dest='threshold', type=float, default=None,
                         help='Find similar documents within above a certain theta')
    parser.add_argument('--preprocess', dest='filter', choices=['LEM', 'STEM'],
                        default=None, help='Articles preprocessing')
    parser.add_argument('--wordcloud', action='store_true', default=False,
                        help='Generate word clouds for each article category')
    parser.add_argument('--classification', choices=['NB', 'RF', 'SVM', 'KNN'],
                        default=None, help='''Runs default classifiers: Naive Bayes, Random Forest,
                        Support Vector Machine and K-Nearest Neighbor''')
    parser.add_argument('--feature', choices=['BoW', 'TF-IDF', 'W2V'], default = None,
                        help='Define features')
    parser.add_argument('--kfold', action='store_true',
                        help='Evaluate and report the performance of each method using 10-fold Cross Validation')
    parser.add_argument('--cache', action='store_true',
                        default=False, help='Use already preprocessed data')
    # return an ArgumentParser object
    return parser.parse_args()

def print_args(args):
    print('---------------------- Applied machine learning in automated text categorization ----------------------\n')
    print("Running with the following configuration")
    # get the __dict__ attribute of args using vars() function
    args_map = vars(args)
    for key in args_map:
        print('\t', key, '-->', args_map[key])
    # add one more empty line for better output
    print()

def main():
    # parse and print arguments
    args = make_args_parser()
    print_args(args)
    # create app object and pass info given by command line arguments
    app = App(args.datasets, args.outputs, args.threshold, args.filter,
              args.wordcloud, args.classification, args.feature, args.kfold, args.cache)
    # run app
    app.run()

if __name__ == '__main__':
    main()
