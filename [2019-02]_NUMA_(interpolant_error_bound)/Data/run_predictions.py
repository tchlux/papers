import os
import numpy as np

from util.stats import cdf_fit
from util.data import Data
from util.system import Timer
from util.math import pairwise_distance
from util.approximate import Delaunay, Voronoi, BoxMesh, ShepMod, \
    LSHEP, NeuralNetwork, MARS, SVR_RBF, class_name

# Make sure all of the prepared and pre-processed data files exist.
from preprocess_data import cwd, raw_dir, data_dir

# Overwrite the print function so that it forces prints to flush.
def print(*args, og_print=print, **kwargs):
    kwargs["flush"] = True
    return og_print(*args, **kwargs)


seed = 0
folds = 10

data_name = lambda p: os.path.basename(p).replace(".dill","").replace(".csv","")

models = [ShepMod, BoxMesh, Voronoi, Delaunay, 
          NeuralNetwork, LSHEP, SVR_RBF, MARS,]

# models = [BoxMesh, SVR_RBF]

# Recording data for each set and for each algorithm that looks like:
# | Data name | Dimension | Fold Size | Fold Number | Model Name | Model fit time |
# | x | ... | y | ... | Model y | ... | Model nearest | Model max edge | Model furthest | Model num contributors | Model prediction time |


intermediate_fit_results = os.path.join(".", "Predictions", "fit-intermediate.dill")
final_fit_results = os.path.join(".", "Predictions", "fit-final.dill")
if os.path.exists(intermediate_fit_results):
    fit_data = Data.load(intermediate_fit_results)
else:
    fit_data = Data(names=["Data name", "Model name", "Train Shape", "Test Shape", "Fold number", "Fit time"],
                    types=[ str,         str,          tuple,         tuple,        int,           float,   ])

# from util.approximate.delaunay import DelaunayP1, DelaunayP2, DelaunayP3
# models = [Delaunay, DelaunayP1, DelaunayP2, DelaunayP3]

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
        # Start a timer, make predictions, record errors.
        for model in models:
            print(f"    running '{class_name(model)}' model...")
            # Check for the viability of this model on this data.
            is_weighted = hasattr(model, "WeightedApproximator")
            if not (is_weighted):
                try:    float(d[0,target])
                except:
                    print("    skipped.")
                    print()
                    continue
            # Initialize the model..
            model = model()

            # Generate the names of the columns where data will be recorded.
            target_guess_col    = f"{class_name(model)} {target}"
            prediction_time_col = f"{class_name(model)} prediction time"
            nearest_col         = f"{class_name(model)} nearest"
            nea_contrib_col     = f"{class_name(model)} nearest contributor"
            fur_contrib_col     = f"{class_name(model)} furthest contributor"
            num_contrib_col     = f"{class_name(model)} number of contributors"
            max_edge_col        = f"{class_name(model)} max edge length"
            # Check to see if the columns are already filled, if so, skip.
            output_columns = (target_guess_col, prediction_time_col, nearest_col)
            if all(c in d.names for c in output_columns):
                if all(None not in test[c] for c in output_columns):
                    print("    all data collected, skipping.")
                    print()
                    continue
            # Initialize the columns of data that will be recorded for
            # this model if that has not been done already.
            if target_guess_col not in d.names:
                d[target_guess_col] = (None for i in range(len(d)))
                d[prediction_time_col] = (None for i in range(len(d)))
                d[nearest_col] = (None for i in range(len(d)))
                if is_weighted:
                    d[nea_contrib_col] = (None for i in range(len(d)))
                    d[fur_contrib_col] = (None for i in range(len(d)))
                    d[num_contrib_col] = (None for i in range(len(d)))
                    d[max_edge_col] = (None for i in range(len(d)))

            # Collect the fit-time data.
            t = Timer()
            model.fit(train_x, train_y)
            t.stop()            
            row = [data_name(data_path), class_name(model),
                   train_x.shape, test_x.shape, i+1, t.total]
            fit_data.append(row)
            fit_data.save(intermediate_fit_results)
            # Now collect the approximation data.
            for test_idx, (d_idx, test_pt) in enumerate(zip(test["Indices"], test_x)):
                if (None not in (d[d_idx,target_guess_col], d[d_idx,prediction_time_col], d[d_idx,nearest_col])):
                    # Skip predictions that have stored results.
                    print(f"    {test_idx+1:4d} :{len(test_x):4d} skipping..", end="\r")                    
                    continue
                print(f"    {test_idx+1:4d} :{len(test_x):4d}           ", end="\r")
                if is_weighted:
                    # Generate the guess based on the weighted sum.
                    t.start()
                    ids, wts = model._predict(test_pt.reshape(1,-1))
                    ids, wts = ids[0], wts[0]
                    guess = sum(train_y[i]*w for (i,w) in zip(ids,wts))
                    t.stop()
                    try: min(wts)
                    except:
                        from util.system import save
                        save((train_x, train_y, test_pt), "bad_box_case.pkl")
                        print("BAD TEST POINT, EXITING")
                        exit()
                    # If some '0' weights were included, remove those.
                    if (min(wts) <= 0):
                        print("     found '0' weights, removing..", end="\r")
                        ids = ids[wts > 0]
                        wts = wts[wts > 0]
                    if (len(ids) == 0):
                        print(f"     failed approximation for data row {d_idx}, skipping..")
                        continue
                    # Identify the nearest contributor.
                    min_dist = float(min(test_train_dists[test_idx, ids]))
                    # Identify the furthest contributor.
                    max_dist = float(max(test_train_dists[test_idx, ids]))
                    # Identify the max edge length.
                    max_edge = float(max(train_dists[ids, ids]))
                    # Record items specific to weighted approximators.
                    d[d_idx, nea_contrib_col] = min_dist
                    d[d_idx, fur_contrib_col] = max_dist
                    d[d_idx, num_contrib_col] = len(wts)
                    d[d_idx, max_edge_col] = max_edge
                else:
                    # Generate the guess using the model.
                    t.start()
                    guess = model(test_pt.reshape(1,-1))
                    t.stop()
                # Store the guess and the total time taken to make the prediction
                try:    d[d_idx, target_guess_col] = float(guess)
                except: d[d_idx, target_guess_col] = guess
                d[d_idx, prediction_time_col] = float(t.total)
                # Calculate the nearest data point in the set and store.
                min_dist = float(min(test_train_dists[test_idx,:]))
                d[d_idx, nearest_col] = min_dist
            # ^^ END (for test point ...)
            # Save intermediate results to a file (after each model).
            d.save(intermediate_results_file)
            print("    done.          ")
            print()
        # ^^ END (for model ...)
    # ^^ END (for fold ...)
    print(f"Saving final results in '{final_results_file}'..")
    print()
    d.save(final_results_file)
# ^^ END (for data ...)
print(f"Saving final fit data file in '{final_fit_results}'..")
print()
fit_data.save(final_fit_results)
