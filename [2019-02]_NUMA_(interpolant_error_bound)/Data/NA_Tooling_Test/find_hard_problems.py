import os
import numpy as np

from util.stats import cdf_fit
from util.data import Data
from util.system import Timer
from util.pairs import pairwise_distance
from util.approximate import Delaunay, Voronoi, BoxMesh, ShepMod, \
    LSHEP, NeuralNetwork, MARS, SVR, class_name

# Make sure all of the prepared and pre-processed data files exist.
from preprocess_data import cwd, raw_dir, data_dir

# Overwrite the print function so that it forces prints to flush.
def print(*args, og_print=print, **kwargs):
    kwargs["flush"] = True
    return og_print(*args, **kwargs)


seed = 0
folds = 10

data_name = lambda p: os.path.basename(p).replace(".dill","").replace(".csv","")

    
count = 0

for raw_file in sorted(os.listdir(raw_dir)):
    if not (("credit" in raw_file) or ("parkinson" in raw_file)): continue

    raw_file = raw_file.replace(".gz","")
    raw_path = os.path.join(raw_dir, raw_file)
    data_path = os.path.join(data_dir, raw_file + '.dill')
    intermediate_results_file = os.path.join(".", "Predictions",
                    f"intermediate_results_{data_name(raw_file)}.dill")
    final_results_file = os.path.join(".", "Predictions",
                    f"results_{data_name(raw_file)}.dill")
    print('-'*70)
    print(data_name(raw_path))
    print(raw_path)
    print(final_results_file)
    print()
    # Check to see if intermediate results have been stored.
    # Load data, declare "target" column, 
    if os.path.exists(intermediate_results_file):
        print("Loading intermediate results..")
        print(intermediate_results_file)
        d = Data.load(intermediate_results_file)
        print(d)
        target = d.names[d.names.index("Indices") - 1]
    else:
        print("Loading data file..")
        d = Data.load(data_path)
        print(d)
        target = d.names[-1]
        # Declare the indices..
        d["Indices"] = range(len(d))
    # Compute the number of training columns.
    num_train_columns = d.names.index(target)
    print(f"Data shape: {d.shape} predicting '{target}'")
    # Start the folds.
    for i,(train, test) in enumerate(d.k_fold(k=folds, seed=seed)):
        print(f"  Fold {i+1} of {folds}..")
        # Get the numeric representation of this data based on training,
        # ignore extra added columns and the "target" column.
        num = train[:,:num_train_columns].to_matrix()
        train_x = num.data
        train_y = list(train[target])
        # Generate the numeric testing set basesd on training.
        test_x = np.array([num.to_real(row[:num_train_columns]) for row in test])
        test_y = list(test[target])
        # Normalize the training and testing data to the unit hypercube.
        train_x = (train_x - num.shift) / num.scale
        test_x  = (test_x  - num.shift) / num.scale
        # Compute distances between test points and trianing points.
        test_train_dists = pairwise_distance(test_x, train_x)
        # Compute the pairwise distance between training points.
        train_dists = pairwise_distance(train_x)

        # Now collect the approximation data.
        for test_idx, (d_idx, test_pt) in enumerate(zip(test["Indices"], test_x)):
            is_credit = ("credit" in raw_file)
            credit_idx = (d_idx in {359, 387, 2666})
            if is_credit and (not credit_idx): continue

            is_park = ("parkinson" in raw_file)
            park_idx = (d_idx in {10,83,714,822,1679,1914,1959,2003,
                                  2067,2081,2531,4699,5355,5756,5795})
            if is_park and (not park_idx): continue

            count += 1

            f_name = f"Hard/hard_problem_{count}.csv"
            print("Training data file", f_name)
            data = np.concatenate((train_x, test_pt[None,:]), axis=0)
            np.savetxt(f_name, data, delimiter=',', comments='',
                       header=f'{train_x.shape[1]},{train_x.shape[0]},1,0')
    # ^^ END (for fold ...)
    print()
# ^^ END (for data ...)
