#!/usr/bin/env python
#
# Just traditional dropout on MNIST as with the others

import varout.experiments
import lasagne.layers
import pickle
import gzip
import os.path

output_file = "no_dropout.pkl.gz"

class DummyDropoutLayer(lasagne.layers.DropoutLayer):
    def __init__(self, incoming, p=0.5, **kwargs):
        # whatever p we get, throw it away
        super(DummyDropoutLayer, self).__init__(incoming, p=0.0)

def main(output_dir, verbose=False):
    # load the data (MNIST)
    for itr in [1,2,3]:
        dataset=varout.experiments.load_dataset()
        # check if there's already a results file
        save_location = os.path.join(output_dir, str(itr) + output_file)
        n = 1
        while os.path.isfile(save_location):
            save_location = os.path.join(output_dir, output_file)+".{0}".format(n)
            n += 1
        # iterate from 100 to 1300 hidden units with 6 points
        # (explicit linspace)
        results = {}
        for n_hidden in [100, 340, 580, 820, 1060, 1300]:
            if verbose:
                print("running {0}".format(n_hidden))
            # make a network with this number of hidden units
            l_out = varout.experiments.srivastavaDropoutArchitecture(
                    DropoutLayer=DummyDropoutLayer,
                    n_hidden=n_hidden)
            # put it in an experiment
            loop = varout.experiments.make_experiment(l_out, dataset)
            # run the experiment with early stopping until it converges
            results[n_hidden] = varout.experiments.earlystopping(loop,
                    verbose=verbose)
        # save the results
        with gzip.open(save_location, "wb") as f:
            pickle.dump(results, f)

if __name__ == "__main__":
    parser = varout.experiments.get_argparser()
    args = parser.parse_args()
    main(args.output_directory, verbose=args.v)
