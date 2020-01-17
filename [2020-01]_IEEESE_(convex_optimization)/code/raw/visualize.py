# Make sure the previous step has been executed.
previous_step = __import__("2-summarize_data")
data_file = previous_step.output_file

import os
from util.data import Data
from util.plot import Plot, PLOT_MARGIN
PLOT_MARGIN = 0


summary_data = Data.load(data_file)
summary_data.sort()

print()
print("summary_data: ",summary_data)

legend = dict(
    xanchor = "center",
    yanchor = "top",
    x = .525,
    y = .92,
    orientation = "h",
    bgcolor="white",
    bordercolor="grey",
    borderwidth=.5
)

layout = dict(
    margin = dict(t=10,b=35,l=42,r=20),
)

no_x_title_layout = dict(
    margin = dict(t=10,b=20,l=42,r=20),
)

# Pixel offset of label from highest value.
label_x_location = 25000
label_y_offset = -1500

# Create a dictionary that returns a default value for missing keys.
class DefaultDict(dict):
    def __init__(self, default=None, *args, **kwargs):
        self.default_value = default
        super().__init__(*args, **kwargs)
    def __getitem__(self, key):
        if (self.default_value is None): return super().get(key, key)
        else: return super().get(key, self.default_value)

# Generate a method for sorting the plots and series.
sort_order = DefaultDict(
    default = DefaultDict(),
    function = DefaultDict(
        default = float('inf'),
        subquadratic = 0,
        superquadratic = 1,
        saddle = 2,
        multimin = 3
    ),
    algorithm = DefaultDict(
        default = float('inf'),
        SGD = 0,
        LBFGS = 1,
        ADAGRAD = 2,
        ADAM = 3,
    )
)
col_name = None
sort_func = lambda key: sort_order[col_name][key]
# Dash style order.
dash_styles = [None, "dot", "dash", "dashdot"]


# Function for making a set of plots given the column and values from
# that column to make plots for as well as the column to use for the
# series in each plot.
def make_plots(plot_col, series_col, plot_values, data, suffix=""):
    # Cycle each plot column value, append plots to one file.
    for plot_col_value in plot_values:
        is_first = plot_col_value == plot_values[0]
        is_last = plot_col_value == plot_values[-1]
        # Get the (sub)set of data for this plot.
        if (plot_col == series_col):
            plot_data = data
            title = ""
        else:
            plot_data = data[ data[plot_col] == plot_col_value ]
            title = f"{str(plot_col_value).title()} {plot_col.title()}"
        # Assign the x-axis title.
        if is_last or (plot_col == series_col): x_axis_title = "Step"
        else:                                   x_axis_title = ""

        # Create the plot.
        p = Plot("", x_axis_title, "Function Value",
                 font_family="times", font_size=17)
        # Create a label for the plot.
        if (plot_col != series_col):
            temp_data = plot_data[ plot_data["step"] == label_x_location ]
            label_height = max(temp_data["90 percentile"])
            del(temp_data)
            label_height += .15*label_height
            if (plot_col_value == "subquadratic"): label_height += .1
            p.add_annotation(title, label_x_location, label_height,
                             ax=0, ay=0, font_family="times",
                             font_size=16, align="center",
                             x_anchor="center", border_width=.5,
                             show_arrow=False, bg_color="#fff")

        # Sort the "series_values".
        global col_name
        col_name = series_col
        series_values = sorted(set(plot_data[series_col]), key=sort_func)
        # Cycle through the series and create them.
        for i,(s,dash) in enumerate(zip(series_values, dash_styles)):
            series_data = plot_data[ plot_data[series_col] == s ]
            c = p.color(i)
            fc = p.color(i, alpha=.08)
            ffc = p.color(i, alpha=.16)
            clear = p.color((0,0,0,0))
            p.add(f"{s} 10th", series_data["step"], series_data["10 percentile"],
                  mode="lines", color=c, group=s, line_width=.1, dash=dash,
                  fill="tonexty", fill_color=fc, show_in_legend=False)
            # p.add(f"{s} 25th", series_data["step"], series_data["25 percentile"],
            #       mode="lines", color=c, group=s, line_width=.1, dash=dash,
            #       fill="tonexty", fill_color=ffc, show_in_legend=False)
            p.add(f"{s}", series_data["step"], series_data["50 percentile"],
                  mode="lines", color=c, group=s, dash=dash,
                  fill="tonexty", fill_color=fc)
            # p.add(f"{s} 75th", series_data["step"], series_data["75 percentile"],
            #       mode="lines", color=c, group=s, line_width=.1, dash=dash,
            #       fill="tonexty", fill_color=fc, show_in_legend=False)
            p.add(f"{s} 90th", series_data["step"], series_data["90 percentile"],
                  mode="lines", color=c, group=s, line_width=.1, dash=dash,
                  show_in_legend=False)

        # Pick the layout according to the existence of axis labels.
        if (len(x_axis_title) == 0): lay = no_x_title_layout
        else:                        lay = layout
        # Create a zoomed in plot (for the same-same plot).
        if (plot_col == series_col):
            p.show(file_name=f"{series_col}-mean-by-{plot_col}-zoomed{suffix}.html",
                   legend=legend, layout=lay, width=500, height=300, 
                   x_range=[0,500], append=False, show_legend=True)
            # p.y_title = ""
        # Generate the usual plot.
        show_legend = is_first
        p.show(file_name=f"{series_col}-mean-by-{plot_col}{suffix}.html",
               legend=legend, layout=lay, width=500, height=300, 
               append=(not is_first), show_legend=show_legend)

        # Exit the for loop after the first iterataion for special case.
        if (plot_col == series_col): break


# Cycle the plot columns and series columns in the summary data.
for (plot_col, series_col, path) in summary_data:
    print()
    print(" plot_col:   ", plot_col)
    print(" series_col: ", series_col)
    print(" path:       ", path)
    
    data = Data.load(path, sample=None)
    print(data)

    # Sort the "plot_values".
    col_name = plot_col
    plot_values = sorted(set(data[plot_col]), key=sort_func)
    plot_values_list = [plot_values]
    while (plot_col != series_col) and (max(map(len, plot_values_list)) > 3):
        for i in range(len(plot_values_list)):
            if len(plot_values_list[i]) > 3:
                break
        else: continue
        # Now 'i' is for an element of plot_values_list with large length.
        #  Pop out that element, break it into two, and reinsert it.
        l = plot_values_list.pop(i)
        plot_values_list.insert(i, l[len(l)//2:])
        plot_values_list.insert(i, l[:len(l)//2])
    # Now every list in "plot_values_list" has length <= 3.
    for i, plot_values in enumerate(plot_values_list):
        suffix = f"-{i}" if i > 0 else ""
        make_plots(plot_col, series_col, plot_values, data, suffix=suffix)

