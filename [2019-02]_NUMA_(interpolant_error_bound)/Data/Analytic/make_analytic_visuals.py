from run_analytic_test import output_data_name
from util.data import Data
from util.plot import Plot, multiplot
from util.stats import cdf_fit
from math import log

# algorithms = {}
algorithms = {"DelaunayP1", "NeuralNetwork"}

BOX_PLOTS = False
MAX_ERROR_LINE = True

if MAX_ERROR_LINE:
    test_func_name = "oscillatory"
    test_size = 1000
    d = None
    for dimension in [2,20]:
        file_name = f"final-{dimension}D-{test_size}-{test_func_name}-analytic.pkl"
        data = Data.load(file_name)
        data.add_column([dimension]*len(data), "Dimension", index=data.names.index("Train"))
        # Generate interesting extra columns.
        data["Abs Errors"] = ([float(abs(v)) for v in l] for l in data["Errors"])
        data["Mean Abs Error"] = (sum(l) / len(l) for l in data["Abs Errors"])
        data["Min Abs Error"] = (min(l) for l in data["Abs Errors"])
        data["Max Abs Error"] = (max(l) for l in data["Abs Errors"])
        if (d is None): d = data
        else:           d += data
    if (len(algorithms) > 0): d = d[d["Algorithm"] == algorithms]
    print(d)
    d.save("all-data.pkl")
    print()
    config_cols = ["Function", "Dimension"]
    configs = d[:,config_cols].unique()
    configs.sort()
    configs.max_display=float('inf')
    print(configs)
    for conf in configs:
        plots = []
        for S in sorted(set(d["SNR"])):
            d_conf = d[d[:,config_cols+["SNR"]] == (list(conf) + [S])]
            F, D = conf
            p = Plot(f"{D}D '{F}' function", f"SNR = {S}", "|<i>error</i>|",
                     font_family="times") #b"log(|<i>error</i>|)")
            seen = {}
            for algorithm in sorted(set(d["Algorithm"])):
                d_alg = d_conf[d_conf["Algorithm"] == algorithm]
                print(d_alg)
                values = []
                locations = []
                # Sort the algorithm data by the 
                d_alg.sort(key=lambda i: i["Train"])
                for row in d_alg:
                    if (row["Train"] < row["Dimension"]): continue
                    locations.append( "N = "+str(row["Train"]) )
                    values.append( [v for v in row["Abs Errors"]] )
                p.add_box(algorithm, values, locations, group=algorithm,
                          show_in_legend=(len(plots) == 0))
                # Save the relevant data to a file.
                d_alg = d_alg.copy()
                d_alg.reorder(["Train", "Abs Errors"])
                d_alg = d_alg[:,:2].copy()
                for er in d_alg["Abs Errors"]: er.sort()
                for i in range(len(d_alg[0,"Abs Errors"])):
                    d_alg[f"abs error {i+1}"] = (er[i] for er in d_alg["Abs Errors"])
                d_alg.pop("Abs Errors")
                d_alg.save(f"{algorithm}-{S}-{F}-{D}.csv")
            # Give the y-axis a log range (like the x axis)
            extra = dict(y_range="", y_axis_settings=dict(type="log"))
            p_fig = p.plot(html=False, layout=dict(boxmode="group"), **extra)
            plots.append( p_fig )
        multiplot( plots, append=True )



# Make box plots
if BOX_PLOTS:
    d = Data.load(output_data_name)
    # Generate interesting extra columns.
    d["Abs Errors"] = ([float(abs(v)) for v in l] for l in d["Errors"])
    d["Mean Abs Error"] = (sum(l) / len(l) for l in d["Abs Errors"])
    d["Min Abs Error"] = (min(l) for l in d["Abs Errors"])
    d["Max Abs Error"] = (max(l) for l in d["Abs Errors"])
    if (len(algorithms) > 0): d = d[d["Algorithm"] == algorithms]
    print(d)
    config_cols = ["Function", "SNR", "Train"]
    configs = d[:,config_cols].unique()
    configs.sort()
    configs.max_display=float('inf')
    print(configs)
    for conf in configs:
        interest = d[d[:,config_cols] == conf]
        interest.sort()
        print(interest)
        # exit()
        F, S, N = conf
        p = Plot(f"{N} train with SNR {S} on '{F}' function")
        seen = {}
        for row in interest:
            name = row["Algorithm"]
            # Skip algorithms that have already been plotted.
            if name in seen: 
                print(f"  skipping extra '{name}'")
                continue
            else:            seen.add(name)
            # Plot the performance of this algorithm.
            fit_time = row["Fit Time"]
            pred_time = row["Predict Time"]
            errors = row["Abs Errors"]
            # f = cdf_fit(errors)
            # p.add_func(name, f, f())
            p.add_box(name, errors)
        p.show(append=True, show=all(conf == configs[-1]))
