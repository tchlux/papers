import os, pickle
import numpy as np
from util.data import read_struct
from util.plotly import Plot, multiplot

DATA_FILE_PKL = "py_data.pkl"
COUNTS_FILE_PKL = "config_counts.pkl"
CLEAN_DATA_PKL = "clean_data.pkl"
DIST_PREDICTION_FILE = "dist_prediction.html"
TEST_HIST_FILE = "tests_by_machine_(aggregated).html"
CLEAN_DATA_CSV = "clean_data.csv"

PATH_TO_CLEAN_DATA = os.path.join("/Users","thomaslux","Git","LargeFiles")
INITIAL_DATA_READ = not os.path.exists(DATA_FILE_PKL)
PLOT_AGGREGATE_HISTOGRAMS = False
VALUES_SUMMARY = True
SHOW_COUNT_SUMMARY = True
SAMPLE_FUNCTION_PREDICTION = False
SAVE_CLEAN_DATA_CSV = False

# ====================================================================
# ====================================================================
#      Read in new data     
if INITIAL_DATA_READ:
    print("Reading data 1...")
    data1 = read_struct(os.path.join(PATH_TO_CLEAN_DATA,"2018-1.csv"))
    print("Reading data 2...")
    data2 = read_struct(os.path.join(PATH_TO_CLEAN_DATA,"2018-2.csv"))
    print("Reading data 3...")
    data3 = read_struct(os.path.join(PATH_TO_CLEAN_DATA,"2018-3.csv"))
    print("Saving all 3 to file...")    
    with open(DATA_FILE_PKL, "wb") as f:
        pickle.dump( (data1, data2, data3), f )
else:
    print("Loading '%s'..."%(DATA_FILE_PKL))
    with open(DATA_FILE_PKL, "rb") as f:
        data = np.concatenate(pickle.load(f))
header = list(data.dtype.names)
print("Data has %i rows. (exclusing 18 erroneous rows)"%(len(data)))
h_width = max(len(h) for h in header)
config_vars = header[2:6] + [header[7]]
print("Variables:",config_vars)

# ====================================================================
# Generate the counts dictionary (pkl contains unique configs and data)
if not os.path.exists(COUNTS_FILE_PKL):
    counts = {}
    size = len(data)
    for i,row in enumerate(data):
        if (not bool(i%100000)): print("%i - %i"%(i,size),end="\r")
        vals = tuple(row[v] for v in config_vars)
        counts[vals] = counts.get(vals, []) + [[row["Throughput"], row["Runtime"]]]
    print("Saving '%s'..."%(COUNTS_FILE_PKL))
    with open(COUNTS_FILE_PKL, "wb") as f:
        for v in counts:
            counts[v] = np.array(counts[v])
        pickle.dump( counts, f )
else:
    print("Loading '%s'..."%(COUNTS_FILE_PKL))
    with open(COUNTS_FILE_PKL, "rb") as f:
        counts = pickle.load(f)

print("="*70)
print(sorted(set(list(map(len, [v for v in counts.values()])))))
exit()
with open("output.txt", "w") as f:
    for config in sorted(counts):
        print("%30s,"%(str(config)[1:-1]),len(counts[config]), file=f)

        # if len(counts[config]) == 2:
        #     print()
exit()

# ====================================================================
# Generate the cleaned data (for prediction and analysis)
if not os.path.exists(CLEAN_DATA_PKL):
    np.random.seed(0)
    clean_data = {}
    for config in sorted(counts.keys()):
        if not (len(counts[config])%150):
            values = counts[config]
            if len(values) > 150:
                # Suffle the values randomly and take the first 150
                np.random.shuffle(values)
                values = values[:150]
            clean_data[config] = values
    print("Saving '%s'..."%(CLEAN_DATA_PKL))
    with open(CLEAN_DATA_PKL, "wb") as f:
        pickle.dump(clean_data, f)
else:
    print("Loading '%s'..."%(CLEAN_DATA_PKL))
    with open(CLEAN_DATA_PKL, "rb") as f:
        clean_data = pickle.load(f)
# ====================================================================
# ====================================================================

if SAVE_CLEAN_DATA_CSV:
    with open(CLEAN_DATA_CSV, "w") as f:
        # Print header
        print(*config_vars,sep=",",end=",", file=f)
        print(*["Trial %i"%(i+1) for i in range(150)],sep=",", file=f)
        for config in sorted(clean_data.keys()):
            print(*config,sep=",",end=",",file=f)
            print(*clean_data[config][:,0],sep=",",file=f)
    print("Done!")


if VALUES_SUMMARY:
    unique = {}
    print()
    for h in header:
        unique_elements = np.unique(data[h])
        if len(unique_elements) < 100:
            unique[h] = sorted(unique_elements)
            print(("%"+str(h_width)+"s")%h,unique[h])
        else:
            print(("%"+str(h_width)+"s")%h,len(unique_elements),"elements values.")
    print()

    if PLOT_AGGREGATE_HISTOGRAMS:
        print("Generating histograms for similarity checks...")
        first = True
        for test in unique["Test"]:
            subset = data[data["Test"] == test]
            machines = sorted(np.unique(subset["Machine"]))
            p1 = Plot("Throughputs by machine for '%s' test"%test,
                      "Throughput", "Probability Mass")
            p2 = Plot("Runtime by machine for '%s' test"%test,
                      "Runtime", "Probability Mass")
            print("","processing test '%s'"%test)
            for m in machines:
                print("","","machine '%s'"%m)
                p1.add_histogram(m,subset[subset["Machine"] == m]["Throughput"],
                                 group=m)
                p2.add_histogram(m,subset[subset["Machine"] == m]["Runtime"],
                                 group=m, show_in_legend=False)

            multiplot([[p1],[p2]], append=(not first), file_name=TEST_HIST_FILE)
            first = False

if SHOW_COUNT_SUMMARY:
    count_array = np.array(list(map(len, counts.values())))
    print("Max: ", np.max(count_array))
    print("Min: ", np.min(count_array))
    print("Mean:", np.mean(count_array))
    print()
    print("# of Unique Configuraitons -- Number of trails")
    for v in sorted(np.unique(count_array)):
        num_configs = sum(count_array == v)
        print("%5s -- %3s trials"%(num_configs,v))
        # if (v % 150):
        num_printed = 0
        for config in counts:
            if len(counts[config]) == v:
                num_printed += 1
                print("        ",config)
            if (num_printed >= 3): 
                print("         sample concluded (there are more)",)
                break

if SAMPLE_FUNCTION_PREDICTION:
    # Test predictability with Delaunay, leave one out, predict
    # distribution using surrounding nodes.
    from util.algorithms import Delaunay
    from scipy.spatial import ConvexHull
    from util.stats import cdf_fit_func, ks_diff, ks_same_confidence

    original_train = np.array([c[:-1] for c in counts if c[-1] == "readers"])
    np.random.seed(10)
    np.random.shuffle(original_train)
    shift = np.min(original_train, axis=0)
    scale = np.max(original_train, axis=0) - shift
    train = (np.asarray(original_train,dtype=np.float64) - shift) / scale

    num_tests = 10
    cv_hull = list(ConvexHull(train).vertices)
    print("Convex hull points:\n","",cv_hull)

    if os.path.exists(DIST_PREDICTION_FILE): 
        os.remove(DIST_PREDICTION_FILE)

    while num_tests > 0:
        ind = np.random.randint(len(train))
        if ind in cv_hull: continue
        num_tests -= 1
        # Build a model and predict the weights / source points with Delaunay
        model = Delaunay()
        model.fit(np.concatenate((train[:ind], train[ind+1:])))
        p_inds, weights = model.points_and_weights(train[ind])
        # Increment the appropriate indices
        p_inds = np.where(p_inds >= ind, p_inds+1, p_inds)
        # Print out some info to the user
        text = "[Frequency, File Size, Record Size, Threads]<br><br>"
        text += "Predicting config: %s<br>"%(tuple(original_train[ind]),)
        text += "Using configs:<br>"

        # Print out the source points and weights, store list of functions
        funcs = []
        for (p, w) in zip(p_inds, weights):
            if ((w-0.00001) > 0):
                orig_point = tuple(original_train[p])
                text += "%5.2f  *  %s<br>"%(w, orig_point)
                d = counts[orig_point + ("readers",)][:,0]
                funcs.append( (cdf_fit_func(d, cubic=True), w) )
        # Get the true function
        true_func = cdf_fit_func(counts[tuple(original_train[ind]) +
                                        ("readers",)][:,0], cubic=True)
        # Generate the new function using the weights
        new_func = lambda x: sum(f(x)*w for (f,w) in funcs)
        # Get the range for the plot
        val_range = [min(counts[tuple(original_train[ind]) +("readers",)][:,0]),
                     max(counts[tuple(original_train[ind]) +("readers",)][:,0]),]
        # Make the plot
        p = Plot("Predicting Throughput Distribution using Delaunay Triangulation",
                 "Throughput Value", "CDF Value (cubic approximation)")
        p.add_func("True function", true_func, val_range)
        p.add_func("Predicted function", new_func, val_range)
        for i,(f,w) in enumerate(funcs):
            p.add_func("Func %i weight %.2f"%(i,w), f, val_range,
                       opacity=0.2, group="sources")

        ks_stat = ks_diff(new_func, true_func)
        n = len(counts[tuple(original_train[ind]) +("readers",)][:,0])
        same_confidence = ks_same_confidence(ks_stat, n)
        text += "KS Stat:    %.2f<br>"%(ks_stat)
        text += "Confidence: %.2e"%(same_confidence)

        x_pos = val_range[0] + (val_range[1]-val_range[0])*0.80
        p.add_annotation(text, x_pos, .05, show_arrow=False,
                         font_family="Courier New, monospace", 
                         font_size=14)
        p.plot(file_name=DIST_PREDICTION_FILE, append=True)
        print()





# Reduce the data to just the clean data, remove all configurations
# that do not have a multiple of 150 runs and reduce those with >150
# runs by random sampling to just 150 runs. Convert this into a 
# 155 x 18k cleaned CSV file.
# 

# All the following code should be general so that I can plug in the
# old and new varsys data.
# 

# Optimize implementation of Voronoi Mesh so that it intelligently
# calculates dot-products.

# Group the data into a format where I can generate functions over
# each of the unique system configurations.
# 

# Do distribution predictions and measure the KS-statistic (and
# another CDF comparison? infinity, 2, and 1 norm differences, as well
# as the signed error distributions)
# 

# Use Delaunay, max box mesh (no bootstrapping), and voronoi mesh (no
# bootstrapping), to make predictions.
# 

# Record the error at all true EDF points for each iteration, present
# a sample distribution of errors seen for one 
# 

# Do a light stastical analysis of the differnces between different
# tests, different threads, frequencies, and file + record sizes.
# 

# Do variance predictions and demonstrate difficulty in predicting
# mean and variance in throughput.
# 

