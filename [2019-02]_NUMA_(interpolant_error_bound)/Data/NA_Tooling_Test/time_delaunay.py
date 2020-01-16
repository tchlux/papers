import os
import numpy as np

from util.data import Data
from util.system import Timer
from util.approximate import Delaunay, class_name

# Make sure all of the prepared and pre-processed data files exist.
from preprocess_data import cwd, raw_dir, data_dir

to_test = 30


# Overwrite the print function so that it forces prints to flush.
def print(*args, og_print=print, **kwargs):
    kwargs["flush"] = True
    return og_print(*args, **kwargs)


seed = 0
folds = 10

data_name = lambda p: os.path.basename(p).replace(".dill","").replace(".csv","")

# Recording data for each set and for each algorithm that looks like:
# | Data name | Dimension | Fold Size | Fold Number | Model Name | Model fit time |
# | x | ... | y | ... | Model y | ... | Model nearest | Model max edge | Model furthest | Model num contributors | Model prediction time |

for raw_file in sorted(os.listdir(raw_dir)):
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
    d = Data.load(data_path)
    target = d.names[-1]
    # Declare the indices..
    d["Indices"] = range(len(d))
    # Compute the number of training columns.
    num_train_columns = d.names.index(target)
    print(f"Data shape: {d.shape} predicting '{target}'")
    # Start the folds.
    for i,(train, test) in enumerate(d.k_fold(k=folds, seed=seed)):
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
        # Start a timer, make predictions, record errors.
        model = Delaunay
        # Initialize the model..
        model = model()
        fit_times = []
        # Collect the fit-time data.
        t = Timer()
        model.fit(train_x, train_y)
        t.stop()            
        # Now collect the approximation data.
        for test_idx, (d_idx, test_pt) in enumerate(zip(test["Indices"], test_x)):
            if test_idx >= to_test: continue
            print(f"    {test_idx+1:4d} :{to_test:4d}           ", end="\r")
            # Generate the guess based on the weighted sum.
            t.start()
            ids, wts = model._predict(test_pt.reshape(1,-1))
            t.stop()
            fit_times.append(t.total)
        # ^^ END (for test point ...)
        # Save intermediate results to a file (after each model).
        print("Median Delaunay fit time:", np.median(fit_times))
        print()
        # Break the "fold" test loop, because this should be constant.
        break
    # ^^ END (for fold ...)
    print()
# ^^ END (for data ...)
