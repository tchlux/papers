from util.plot import Plot
from util.data import Data
from util.stats import cdf_fit


raw_results_file = "[2]-results.csv.gz"
res = Data.load(raw_results_file, sample=None)
algs = sorted(set(res["fit"]))
algs = [("EDF", "flat_fit"), ("Linear", "linear_fit"),
        ("Cubic", "cubic_fit"), ("Quintic", "quintic_fit")]

def make_samp_plot(samp_size):
    p = Plot("","Absolute Error", "CDF", font_family="times", font_size=16)

    styles = [None, "dashdot", "dot", "dash"]

    d = res[res["sample"] == samp_size]
    print("ten: ",d)
    for (n,a),s in zip(algs,styles):
        print("a: ",a)
        subdata = d[d["fit"] == a][:,3:]
        print("subdata: ",subdata)
        # Get the distribution of the absolute values.
        values = [abs(v) for row in subdata for v in row]
        if (len(values) == 0): continue
        abs_cdf = cdf_fit( values )
        p.add_func(n, abs_cdf, [0, .5], dash=s)

    # Set the legend properties.
    legend = dict(
        xanchor = "center",
        yanchor = "top",
        x = .71,
        y = .8,
        orientation = "v",
        bgcolor="white",
        bordercolor="grey",
        borderwidth=.5
    )

    p.show(file_name=f"abs-errors-{samp_size}-samples.html",
           width=400, height=220, legend=legend)


make_samp_plot(10)
make_samp_plot(50)
make_samp_plot(200)
