from util.plot import Plot, multiplot
from util.stats import cdf_fit_func
import numpy as np
from util.data import Data

# d = Data.load("knn_results_[yelp-mnist].pkl")
d = Data.load("prediction_results.pkl")
names = d.names.copy()
names = [names[0], names[6], names[1]] + names[2:6] + names[7:]
d.reorder( [names[0], names[6], names[1]] )
d.sort()
d["Errors"] = ([float(v) if (v <= 5) else 5.0 for v in e] for e in d["Errors"])
d["Mean Error"] = (sum(e) / len(e) for e in d["Errors"])
d["Mean Squared Error"] = (sum(v**2 for v in e) / len(e) for e in d["Errors"])
d["Error Variance"] = (float(np.var(e)) for e in d["Errors"])

d._max_display = 1000
print(d)
all_data = d

# Get the unique dimensions and algorithms.
dims = sorted(set(all_data["Dimension"]))
algs = sorted(set(all_data["Algorithm"]))
data_sets = sorted(set(all_data["Data"]))

for ds in data_sets:
    for alg in algs:
        d = all_data[all_data["Data"] == ds]
        d = d[d["Algorithm"] == alg]
        min_index = int(np.argmin(d["Mean Error"]))
        m = d[min_index, "Method"]
        e = d[min_index, "Mean Error"]
        d = str(d[min_index, "Dimension"])
        print(" | ".join(["",ds, alg, m, d, f"{e:.3f}",""])[1:])
print()



# PCA versus MPCA-10
colors = {}
plots = []
for data_name in ("yelp", "mnist", "cifar"):
    for sample_ratio in (1,):
        for a in algs:
            # Skip uninteresting sets.
            if (data_name == "yelp") and (a != "KNN10"): continue
            if (data_name == "mnist") and (a != "KNN1"): continue
            if (data_name == "cifar") and (a != "KNN10"): continue
            # Break up data by dimension
            y_axis = "Count" if data_name == "yelp" else ""
            p = Plot("", data_name +" errors", y_axis, font_size=20, font_family="times")
            plots.append(p)
            for method in ("MPCA", "PCA"):
                for dim in dims:
                    # Color by method
                    if method not in colors: colors[method] = p.color(len(colors))
                    color = colors[method]
                    # Reduce data
                    d = all_data[all_data["Data"] == data_name]
                    d = d[d["Method"] == method]
                    d = d[d["Sample Ratio"] == sample_ratio]
                    d = d[d["Algorithm"] == a]
                    d = d[d["Dimension"] == dim]
                    # Skip empty data sets.
                    if len(d) == 0: continue

                    # cdf = cdf_fit_func(d[0,"Errors"])
                    # p.add_func(method+"-"+a+"-"+str(dim), cdf, cdf(), group=d)
                    plot_name = method #+"-"+a+"-"+str(dim)
                    show = (dim == dims[0]) and (data_name == "yelp")
                    if data_name == "yelp":
                        start_end = (-.25, 5.25)
                        num_bins = 11
                        p.add_histogram(plot_name, d[0,"Errors"], show_in_legend=show,
                                        num_bins=num_bins, marker_line_width=10,
                                        marker_line_color="rgb(0,0,0)",
                                        start_end=start_end, padding=0.0,
                                        barmode="", color=color)
                    else:
                        num_bins = 2
                        start_end = (-.5, 1.5)
                        p.add_histogram(plot_name, d[0,"Errors"], show_in_legend=show,
                                        num_bins=num_bins, barmode="",
                                        color=color, marker_line_width=10,
                                        marker_line_color="rgb(0,0,0)",
                                        start_end=start_end, padding=0.0)
            # p.show(append=True)


legend_settings = dict(
    xanchor = "center",
    yanchor = "top",
    x = .5,
    y = 1.25,
    bordercolor = "#DDD",
    borderwidth = 1,
    font = dict(size=15),
    orientation = "h",
)

plots = [p.plot(legend=legend_settings, show=False) for p in plots]
multiplot(plots, width=700, height=150, gap=.06,
          file_name="errors_histogram.html")

