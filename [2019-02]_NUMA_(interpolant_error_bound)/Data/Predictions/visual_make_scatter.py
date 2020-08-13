
import os, sys

old_path = sys.path
sys.path = [os.path.abspath(os.curdir)] + old_path
from util.data import Data
from util.plot import Plot

results = ["results_iozone_150.dill",       
           "results_forestfires.dill",      
           "results_parkinsons_updrs.dill", 
           "results_weatherAUS.dill",       
           "results_creditcard.dill",       
]

targets = {
    "forestfires":("area", ["SVR", "BoxMesh", "MARS"], True),
    "parkinsons":("total_UPDRS", ["ShepMod", "NeuralNetwork", "Voronoi"], False),
    "weather":("Rainfall Tomorrow", ["NeuralNetwork", "LSHEP", "MARS"], True),
    "creditcard":("Amount", ["NeuralNetwork", "Delaunay", "ShepMod"], True),
}


legend = dict(
    xanchor = "center",
    yanchor = "top",
    x = .5,
    y = .95,
    orientation = "h",
    bgcolor="white",
    bordercolor="grey",
    borderwidth=.5,
    traceorder='reversed'
)
ms = 5 # Marker size

all_symbols = ["circle", "diamond", "cross", "x",
               "star", "pentagon", "diamond-wide",
               "star-triangle-down"]
all_colors = [(126,153,210), (110,200,109), (210,105,109),
              (200,153,200), (255,180,40), (110,200,200),
              (150,150,180), (255,180,40)]

all_names = set(sum((v[1] for v in targets.values()),[]))
all_names = sorted(all_names, key=lambda n: -sum(n in v[1] for v in targets.values()))
all_names = [n.replace("NeuralNetwork","MLP")\
              .replace("BoxMesh", "BoxSpline") for n in all_names]

all_symbols = {n:s for (n,s) in zip(all_names,all_symbols)}
all_colors = {n:c for (n,c) in zip(all_names,all_colors)}
all_colors["Voronoi"], all_colors["LSHEP"] = all_colors["LSHEP"], all_colors["Voronoi"]
all_symbols["SVR"], all_symbols["LSHEP"] = all_symbols["LSHEP"], all_symbols["SVR"]
print("all_names: ",all_names)
print("all_symbols: ",all_symbols)
print("all_colors: ",all_colors)


for r_path in results:
    for t in targets:
        if t in r_path: break
    else: continue
    print()
    print("r_path: ",r_path)
    print("target: ",t)
    d = Data.load(r_path)
    target, models, log_scale = targets[t]
    col_names = [target] + [m+" "+target for m in models]
    d = d[col_names]
    print(d)
    # Compute the mininum and maximum values.
    min_max = [min(d[:,0]), max(d[:,0])]
    print("min_max: ",min_max)
    
    target = target.replace("_", " ").title()\
                                     .replace("Updrs","UPDRS")\
                                     .replace("Area", "Area Burned")\
                                     .replace("Amount", "Transaction Amount")
    ph = Plot("Distribution of absolute errors", target)

    p = Plot("", f"Actual {target}", f"Predicted {target}",
             font_family="times", font_size=18)
    # The series will be displayed in order, they are only plotted backwards for layering.
    # Construct a histogram for sanity checking.
    ph.add_histogram("Data", list(d[:,0]), color=(0,0,0))
    # Loop over all the series by name.
    for i,n in enumerate(d.names[1:][::-1][:3]):
        print("n: ",n)
        i = 3 - i - 1
        series_name = n.split()[0]\
                       .replace("NeuralNetwork","MLP")\
                       .replace("BoxMesh", "BoxSpline")
        s = all_symbols[series_name]
        c = all_colors[series_name]
        # Add the black background (border)
        p.add(series_name, d[:,0], d[n], symbol=s, color=(0,0,0),
              marker_size=ms+1, show_in_legend=False)
        # Add the actual series.
        p.add(series_name, d[:,0], d[n], symbol=s, color=c,
              marker_size=ms)
        # Compute the errors.
        errors = [t-g for t,g in zip(d[:,0], d[n])]
        ph.add_histogram(series_name, list(map(abs,errors)), color=c)
    # Compute the plotting range.
    import math
    # Set the axis type.
    if log_scale:
        axis_settings = dict(type="log")
        if (min_max[0] <= 0): min_max[0] = 0.1
        min_max = [math.log(min_max[0],10), math.log(min_max[1],10)]
    else: axis_settings = {}
    # Make the plot.
    p.plot(file_name=f"visual_scatter_plots_{t}.html",
           width=450, height=450, x_range=min_max, y_range=min_max,
           y_axis_settings=axis_settings,
           x_axis_settings=axis_settings,
           legend=legend,
           show=True)

    # ph.show(x_axis_settings=axis_settings)
