#!/usr/bin/env python
#
# Just traditional dropout on MNIST as with the others

import varout.experiments
import varout.objectives
import varout.layers
import pickle
import gzip
import os.path

output_file = "variationalA.pkl.gz"

def main(output_dir, verbose=False):
    # load the data (MNIST)
    for itr in [1,2,3,4,5]:
        dataset=varout.experiments.load_dataset()
        #dataset = varout.experiments.load_cifar10()
        # check if there's already a results file
        save_location = os.path.join(output_dir, str(itr) + output_file)
        n = 1
        while os.path.isfile(save_location):
            save_location = os.path.join(output_dir, output_file)+".{0}".format(n)
            n += 1
        # iterate from 100 to 1300 hidden units with 6 points
        # (explicit linspace)
        results = {}
        for n_hidden in [100, 340, 580, 820, 1060]:
            if verbose:
                print("running {0}".format(n_hidden))
            # make a network with this number of hidden units
            l_out = varout.experiments.vardropADropoutArchitecture(
                    n_hidden=n_hidden)
            # put it in an experiment
            loop = varout.experiments.make_experiment(l_out, dataset,
                    extra_loss=-varout.objectives.priorKL(l_out)/50000)
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
