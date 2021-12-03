"""Run benchmark on the tree booster."""

import argparse
import ast
import time
import dpctl
import pdb

import numpy as np
import xgboost as xgb

RNG = np.random.RandomState(1994)


def run_benchmark(args, tree_alg):
    """Runs the benchmark."""
    try:
        dtest = xgb.DMatrix('dtest.dm')
        dtrain = xgb.DMatrix('dtrain.dm')

        if not (dtest.num_col() == args.columns
                and dtrain.num_col() == args.columns):
            raise ValueError("Wrong cols")
        if not (dtest.num_row() == args.rows * args.test_size
                and dtrain.num_row() == args.rows * (1 - args.test_size)):
            raise ValueError("Wrong rows")
    except:
        print("Generating dataset: {} rows * {} columns".format(args.rows, args.columns))
        print("{}/{} test/train split".format(args.test_size, 1.0 - args.test_size))
        tmp = time.time()
        X = RNG.rand(args.rows, args.columns)
        y = RNG.randint(0, 2, args.rows)
        if 0.0 < args.sparsity < 1.0:
            X = np.array([[np.nan if RNG.uniform(0, 1) < args.sparsity else x for x in x_row]
                          for x_row in X])

        train_rows = int(args.rows * (1.0 - args.test_size))
        test_rows = int(args.rows * args.test_size)
        X_train = X[:train_rows, :]
        X_test = X[-test_rows:, :]
        y_train = y[:train_rows]
        y_test = y[-test_rows:]
        print("Generate Time: %s seconds" % (str(time.time() - tmp)))
        del X, y

        tmp = time.time()
        print("DMatrix Start")
        dtrain = xgb.DMatrix(X_train, y_train, nthread=-1)
        dtest = xgb.DMatrix(X_test, y_test, nthread=-1)
        print("DMatrix Time: %s seconds" % (str(time.time() - tmp)))
        del X_train, y_train, X_test, y_test

        dtest.save_binary('dtest.dm')
        dtrain.save_binary('dtrain.dm')

    param = {'objective': 'binary:logistic'}
    if args.params != '':
        param.update(ast.literal_eval(args.params))

    #pdb.set_trace()
    if (tree_alg == "hist"):
        param['tree_method'] = tree_alg
        print("Training with tree method '%s'" % param['tree_method'])
    else:
        param['updater'] = tree_alg

    param['tree_method'] = args.tree_method
    param['device_id'] = args.device_id
    print("Training with '%s'" % param['tree_method'])
    tmp = time.time()
    xgb.train(param, dtrain, args.iterations, evals=[(dtest, "test")])
    print("Train Time: %s seconds" % (str(time.time() - tmp)))


def main():
    """The main function.

    Defines and parses command line arguments and calls the benchmark.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--updater', default='grow_quantile_histmaker_oneapi')
    parser.add_argument('--device_id', default=1)
    parser.add_argument('--tree_method', default='hist')
    parser.add_argument('--sparsity', type=float, default=0.0)
    parser.add_argument('--rows', type=int, default=1000000)
    parser.add_argument('--columns', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--test_size', type=float, default=0.25)
    parser.add_argument('--params', default='',
                        help='Provide additional parameters as a Python dict string, e.g. --params '
                             '\"{\'max_depth\':2}\"')
    args = parser.parse_args()


    device_list = dpctl.get_devices() #Returns a list of class <class 'dpctl._sycl_device.SyclDevice'>
    num_runs = 3
    #tree_alg = ['hist', 'grow_quantile_histmaker', 'grow_quantile_histmaker_oneapi']
    tree_alg = ['grow_quantile_histmaker_oneapi']
    
    #pdb.set_trace()
    for i in device_list:
        print("Current device: " + i.name)
        for j in tree_alg:
            print("Using this tree algorithm: " + j)
            for k in range(1,num_runs+1):
                print("Run number " + str(k))
                run_benchmark(args, j) #Need to pass updater, tree method, device id; which are all passed in args


if __name__ == '__main__':
    main()
