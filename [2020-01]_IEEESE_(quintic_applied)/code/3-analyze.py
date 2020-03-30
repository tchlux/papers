import os
from numpy import mean
from util.data import Data
from util.plot import Plot
from util.stats.rank import rank_probability

# --------------------------------------------------------------------
# d = Data.load("processed-results.csv", sample=None)
# print(d)
# d.save("processed-results.pkl")
# --------------------------------------------------------------------

processed_results_file = "[2]-processed-results.pkl"
# Generate the processed results file (in pickle format).
if not os.path.exists(processed_results_file):
    d = Data.load(processed_results_file[:-4] + ".csv.gz", sample=None)
    d.save(processed_results_file)
else:
    d = Data.load(processed_results_file)

# Get rid of the trial counter.
d.pop("trial")
# Stack all the computed summary data.
d.stack(d.names[2:])

# Compute the Rank 0 probabilities for all the columns.
samples = sorted(set(d["sample"]))
algs = sorted(set(d["fit"]))
for col in d.names[2:]:
    # Initialize the column to have None values everywhere.
    d[col + " rank 0"] = 0.0
    # Cycle through sample sizes.
    for s in samples:
        s_data = d[d["sample"] == s]
        # Cycle through algorithms (picking one versus others).
        for a in algs:
            mine = s_data[s_data["fit"] == a]
            others = s_data[s_data["fit"] != a]
            mine[col + " rank 0"] = rank_probability(
                mine[0,col], list(others[col]), rank=0)

samples = sorted(set(d['sample']))
algs = [("EDF", "flat_fit"), ("Linear", "linear_fit"),
        ("Cubic", "cubic_fit"), ("Quintic", "quintic_fit")]

p = Plot("", "Samples", "KS Rank 0 Probability", font_family="times", font_size=16)
styles = [None, "dashdot", "dot", "dash"]
for i,((n,a),s) in enumerate(zip(algs,styles)):
    alg_data = d[d["fit"] == a]
    x = list(alg_data["sample"])
    for col in d.names[-3:]:
        if "KS" not in col: continue
        y = list(alg_data[col])
        p.add(n, x, y, mode="markers+lines", color=p.color(i), dash=s)
        # p.add(n + " " + col.split()[0], x, y, mode="markers+lines",
        #       color=p.color(i+1), group=col)

# Set the legend properties.
legend = dict(
    xanchor = "center",
    yanchor = "top",
    x = .6,
    y = .25,
    orientation = "h",
    bgcolor="white",
    bordercolor="grey",
    borderwidth=.5
)
p.show(file_name="KS-rank-0-probability.html", width=800, height=300, legend=legend)


