#!/usr/bin/env python
#
# Just traditional dropout on MNIST as with the others

import varout.experiments
import varout.objectives
import varout.layers
import pickle
import gzip
import os.path
from scipy.io import loadmat

output_file = "effect_variationalA_svhn.pkl.gz"
def load_data(path):
    """ Helper function for loading a MAT-File"""
    data = loadmat(path)
    for i in range(len(data['y'])):
        if data['y'][i]==10:
            data['y'][i] =0
    return data['X']/255.0, data['y']


def load_svhn():
    train_x, train_y = load_data('train_32x32.mat')
    test_x, test_y = load_data('test_32x32.mat')
    train_x, train_y = train_x.transpose((3, 2,0, 1)), train_y[:, 0]
    test_x, test_y = test_x.transpose((3, 2,0, 1)), test_y[:, 0]

    return dict(X_train=train_x,
                y_train=train_y,
                #X_valid=X_val.reshape(-1, 784),
                #y_valid=y_val,
                X_test=test_x,
                y_test=test_y)

def main(output_dir, verbose=False):
    # load the data (MNIST)
    for itr in [3]:
        dataset=load_svhn()
        # check if there's already a results file
        save_location = os.path.join(output_dir, str(itr) + output_file)
        n = 1
        while os.path.isfile(save_location):
            save_location = os.path.join(output_dir, output_file)+".{0}".format(n)
            n += 1
        # iterate from 100 to 1300 hidden units with 6 points
        # (explicit linspace)
        results = {}
        for n_hidden in [1.5,2]:
            if verbose:
                print("running {0}".format(n_hidden))
            # make a network with this number of hidden units
            l_out = varout.experiments.Effect_vardropADropoutArchitecture_cifar( batch_size=100,
                    n_hidden=n_hidden)
            # put it in an experiment
            loop = varout.experiments.make_experiment_svhn(l_out, dataset, batch_size=100,
                    extra_loss=-varout.objectives.GaussianKL(l_out)/73257.0)
            # run the experiment with early stopping until it converges
            results[n_hidden] = varout.experiments.earlystopping_cifar(loop,
                    verbose=verbose)
        # save the results
        with gzip.open(save_location, "wb") as f:
            pickle.dump(results, f)

if __name__ == "__main__":
    parser = varout.experiments.get_argparser()
    args = parser.parse_args()
    main(args.output_directory, verbose=args.v)
