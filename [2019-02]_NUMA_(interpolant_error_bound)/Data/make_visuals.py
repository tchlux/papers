import os, math
import numpy as np
from util.plot import Plot
from util.data import Data
from util.math import flatten, transpose
from util.stats import cdf_fit, rank_probability
from util.misc.paper import latex_table


percentiles = [0, 25, 50, 75, 100]

model_order = [
    # "BFGS1000",
    # "SGD10K",
    "MARS",
    "SVR_RBF",
    "NeuralNetwork",
    "Delaunay",
    "ShepMod",
    "LSHEP",
    "Voronoi",
    "BoxMesh",
]

data_order = ["results_forestfires.dill",
              "results_parkinsons_updrs.dill",
              "results_weatherAUS.dill",
              "results_creditcard.dill",
              "results_iozone_150.dill",
]


print("Importing preprocess data..",end="\r")

# Make sure all of the prepared and pre-processed data files exist.
from preprocess_data import cwd, raw_dir, data_dir, results_folder, processed_results_folder

print()

fit_time = Data.load(os.path.join(results_folder,"fit-final.dill"))

# Create a container for all of the 'best' percents and 'timings'
all_comparisons = {}

# Cycle through and prepare data from results.
for f in data_order:
    if ("results" != f[:len("results")]): continue
    # Work with the other data.
    data_name = f[len("results_"):-len(".dill")]
    path = os.path.join(results_folder, f)
    print('\n' + '-'*70)
    print(f"loading {data_name}..", end="\r")
    d = Data.load(path)
    print(f, data_name, d.shape)

    print("extracting fit data..", end="\r")
    # Get the "fit" data from the records.
    sub_fit_data = fit_time[fit_time["Data name"] == data_name].copy()
    sub_fit_data.reorder([sub_fit_data.names[0],"Model name"])
    # Replace the fit times with average fit time per training point.
    timing_data = sub_fit_data[:,:2].unique().copy().collect(sub_fit_data)
    timing_data['Fit time'] = map(float,map(np.median, timing_data['Fit time']))
    for n in timing_data.names[2:-1]:
        timing_data.pop(n)
    timing_data['Point Predict time'] = (None for row in timing_data)
    timing_data['Total Predict time'] = (None for row in timing_data)
    timing_data._max_display = len(timing_data)

    print("extracting timing data..", end="\r")
    # Get the target column and the list of models that made predictions.
    target_index = d.names.index("Indices") - 1
    target = d.names[target_index]
    models = [c.split()[0] for c in d.names[target_index+2:] if target in c][::-1]
    # Generate a plot of the model errors.
    p = Plot(f"{f} Absolute Error Distributions")
    max_error = -float('inf')
    table = [["Algorithm", "Min"] + [f"${v}^{{th}}$" for v in percentiles[1:-1]] + ["Max"]]
    # Remove those models that are not supposed to appear.
    models = [m for m in models if m in model_order]
    # Re-order the timing data to reflect the desired model order.
    indices = []
    for m in model_order:
        if m not in models: continue
        indices.append( list(timing_data["Model name"]).index(m) )
    timing_data = timing_data[indices].copy()

    print("collecting error data..", end="\r")
    # Collect the error and timing data.
    error_data = Data()
    for i,m_name in enumerate(sorted(models, key=lambda m: model_order.index(m))):
        # Collect errors for all rows that have valid predictions for all models
        errors = []
        prediction_times = []
        for row in range(len(d)):
            if 'iozone' not in data_name:
                to_clean = [f"{m} {target}" for m in models]
                if "nan" in set(map(str,d[row,to_clean])): continue
            truth, guess = d[row, target], d[row, f"{m_name} {target}"]
            errors.append( truth - guess )
            prediction_times.append(d[row, f"{m_name} prediction time"])
        # Store the median prediction time for this model on this data.
        timing_data[i,"Point Predict time"] = float(np.median(prediction_times))
        timing_data[i,"Total Predict time"] = timing_data[i,"Point Predict time"]*(len(d)//10)
        # Skip bad models.
        if 'nan' in map(str,errors):
            print(f"  {m_name} had 'nan' values in its predictions..")
            errors = [v for v in errors if str(v) != 'nan']
        if (len(errors) == 0):
            print(f"  skipping {m_name} because no predictions are found..")
            continue
        abs_errors = list(map(float,map(abs,errors)))
        error_data.append( [m_name.replace("NeuralNetwork","MLP")
                            .replace("BoxMesh","BoxSplines")
                            .replace("SVR_RBF","SVR")] + abs_errors )
        max_error = max(max_error, max(abs_errors))

        # p.add_box(m_name, abs_errors)
        table.append([m_name] +[np.percentile(abs_errors, p) for p in percentiles])

    print("  saving error data and timing data..", end="\r")
    error_data.names[0] = "Algorithm"
    error_data.save(os.path.join(processed_results_folder,"absolute_error_" + f[:-4] + "csv"))

    # Fix model names in timing data.
    for i in range(len(timing_data)):
        timing_data[i,"Model name"] = timing_data[i,"Model name"] \
                                      .replace("NeuralNetwork", "MLP") \
                                      .replace("BoxMesh", "BoxSplines") \
                                      .replace("SVR_RBF", "SVR")
    timing_data.save(os.path.join("Results","timing_data_" + f[:-4] + "csv"))

    # Generate the table for the errors.
    t = latex_table(table[1:], table[0],
                    side_bars=False, fmt=".2e", decimal_align=True, #simplify_sig=6,
                    **dict([(f"wrap_{i}_0", ("\\mathbf{","}"))
                            for i in range(len(table[0]))] +
                           [(f"wrap_{i}_1", ("\\mathit{","}"))
                            for i in range(len(table[0]))])) \
                                .replace("NeuralNetwork", "MLP") \
                                .replace("BoxMesh", "BoxSpline") \
                                .replace("SVR_RBF", "SVR")
    print(' '*70)
    print(t)
    print()

    # Generate table of "% best" and "fit time".
    models = list(error_data["Algorithm"])
    table = []
    best_sum = 0
    scnd_wrst_sum = 0
    wrst_sum = 0
    for i,m in enumerate(models):
        other_data = list(range(len(error_data)))
        this_data = [other_data.pop(i)]
        # Extract the actual values.
        this_data = error_data[i][1:]
        other_data = [error_data[idx][1:] for idx in other_data]
        # Get the rank probabilities for being the best and the worst.
        best = rank_probability(this_data, other_data, 0, order=True)
        scnd_wrst = rank_probability(this_data, other_data, -2, order=True)
        wrst = rank_probability(this_data, other_data, -1, order=True)
        # # Print out those probabilities.
        # print(f"  {m:14s} %.2e %.2e"%(best, wrst))
        # Track the total probability displayed (for sanity check).
        best_sum += best
        scnd_wrst_sum += scnd_wrst
        wrst_sum += wrst
        # Append the information to the table.
        table.append([m, best*100, "",
                      timing_data[i,"Fit time"],
                      timing_data[i,"Point Predict time"],
                      timing_data[i,"Total Predict time"]])

        if 'iozone' not in data_name:
            all_comparisons[m] = all_comparisons.get(m,[]) + [tuple(table[-1])]

    # Compute the sizes for the caption.
    fit_size = d.shape[0] - d.shape[0] // 10
    predict_size = d.shape[0] // 10

    formats = ['', ".1f", ''] + [".2e"]*3
    wrappers = ['', ("$","$"), ''] + [("$","$")]*3

    # Make the table for %best
    t = latex_table(table, ['Algorithm', '\% Best', "", "Fit / Prep. Time", "App. Time (s)", "Total App. Time"],
                    side_bars=False, fmt=formats, wrap_nums=wrappers, #simplify_sig=6,
                    decimal_align=True,
                    **{"wrap_1_-1" : ("\\mathbf{","}"),
                       "wrap_1_-2" : ("\\mathit{","}"),
                       "wrap_3_0" : ("\\mathbf{", "}"),
                       "wrap_3_1" : ("\\mathit{", "}"),
                       "wrap_4_0" : ("\\mathbf{", "}"),
                       "wrap_4_1" : ("\\mathit{", "}"),
                       "wrap_5_0" : ("\\mathbf{", "}"),
                       "wrap_5_1" : ("\\mathit{", "}"),}
    ).replace('caption{}', f'caption{{{data_name} fit size {fit_size}, predict size {predict_size}.}}')\
    .replace("\\hline", "\\cline{1-2}\\cline{4-6}") \
    .replace("NeuralNetwork", "MLP").replace("BoxMesh", "BoxSpline").replace("SVR_RBF","SVR")
    print(' '*70)
    print(t)
    print()

    # If this is I/O zone, generate table for different KS stats.
    if "iozone" in data_name:
        header = '\\textbf{Algorithm}', '\\textbf{$P$-Value}', '\\textbf{N.H. Rejections}'
        table = []
        ks_values = [0.1568, 0.1879, 0.2251, 0.3110]
        p_values = [".05", ".01", ".001", "1.0e-6"]
        models = [''] + list(error_data["Algorithm"])
        columns = [models]
        for (ks,p) in zip(ks_values, p_values):
            rejections = [f"$p$ = {p}"]
            for row in error_data:
                model, errors = row[0], np.array(row[1:])
                models.append(model)
                rejections.append( 100 * sum(errors > ks) / len(errors) )
            columns.append( rejections )
        table = transpose(columns)
        t = latex_table(table[1:], table[0], fmt=".1f",
                        wrap_nums=("$", "\%$"), **dict(
            [(f"wrap_{i}_0", ("\\mathbf{","}")) for i in (1,2,3,4)] +
            [(f"wrap_{i}_1", ("\\mathit{","}")) for i in (1,2,3,4)]))
        print(' '*70)
        print(t)
        print()



print()
print()
print("Saving comparison data..")
from util.system import save
save(all_comparisons, "all_comparisons_data.pkl")
print()
print("Loading comparison data..")
from util.system import load
all_comparisons = load("all_comparisons_data.pkl")
for m in all_comparisons:
    _, best, _, fit, one, many = list(zip(*all_comparisons[m]))
    all_comparisons[m] = tuple(map(np.mean, (best, fit, one)))

table = []
for m in model_order:
    m = m.replace("NeuralNetwork","MLP").replace("BoxMesh","BoxSplines").replace("SVR_RBF","SVR")
    table.append([m] + list(all_comparisons[m]))
    # print(f"{m:13s}", all_comparisons[m])

formats = ['', ".1f"] + [".0e"]*2
wrappers = ['', ("$","$")] + [("$","$s")]*2

# Make the table.
t = latex_table(table, ['Algorithm', 'Avg. \% Best', "Avg. Fit Time", "Avg. App. Time (1)"],
                side_bars=False, fmt=formats, wrap_nums=wrappers, decimal_align=True,
                **{"wrap_1_-1" : ("\\mathbf{","}"),
                   "wrap_1_-2" : ("\\mathit{","}"),
                   "wrap_2_0" : ("\\mathbf{", "}"),
                   "wrap_2_1" : ("\\mathit{", "}"),
                   "wrap_3_0" : ("\\mathbf{", "}"),
                   "wrap_3_1" : ("\\mathit{", "}"),})

print(' '*70)
print(t)
print()

    


exit()


# FOR EACH DATA SET

# Absolute error box plot
# Absolute error quartiles table with bold minimum values
# Stacked bar chart per-point fit & predict times


# Absolute error distribution [(model) x (data set)].
# Absolute error distribution versus distance to nearest neighbor.
# Absolute error distribution versus diameter of influencers.
# Histogram of number of contributors for each and model.
# Table of fit times for each model.

# Box plot distribution of prediction time by dimension (and by number points)
