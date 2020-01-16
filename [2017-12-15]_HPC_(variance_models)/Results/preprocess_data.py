import os, pickle
from util.plotly import Plot, multiplot
from util.data import read_struct
from util.paper import latex_table
import numpy as np


# ==================================
#      Preparing the data files     
# ==================================

all_data_file = "all_data.pkl"

algorithms = ["LSHEP", "MARS", "MLPRegressor", "qHullDelaunay", "SVR"]

if not os.path.exists(all_data_file):
    predictors = ["Mean", "Variance"]
    # Process all raw files into pkl files that hold the contents (for
    # easier loading later on)
    data = tuple()
    for p in predictors:
        summary_output = p+"_all_summary.pkl"
        all_data = None
        for a in algorithms:
            source = "Output-%s_%s-MDA_results.csv"%(a,p)
            a_data = read_struct(source)
            print(len(a_data), source)
            if type(all_data) == type(None):
                all_data = a_data
            else:
                all_data = np.concatenate((all_data, a_data))
        print("All collected:")
        print(len(all_data))
        data += (all_data,)

    # Save all data to file
    with open(all_data_file, "wb") as f:
        pickle.dump(data, f)
    # Store locally with different variable names
    mean_data, var_data = data
        
else:
    # Load all data from file
    with open(all_data_file, "rb") as f:
        (mean_data, var_data) = pickle.load(f)


# ===================================
#      Processing the data files     
# ===================================

print()
total_printed = 0
for h in mean_data.dtype.names:
    total_printed += len(str(h) + "   ")
    if total_printed >= 70:
        print()
        total_printed = len(str(h) + "   ")
    print(str(h),end="   ")
print()
print()


#      High level statistics     
# ===============================

#      High Level Algorithm Performance     
# ==========================================
for a in algorithms:
    mean_error = mean_data[mean_data["Algorithm"] == a]["Relative_Mean_Error"]
    var_error = var_data[var_data["Algorithm"] == a]["Relative_Mean_Error"]
    print("%s:"%a, len(mean_error), len(var_error))
    print("","Mean Error  ", mean_error.mean(), var_error.mean() )
    print("","Median Error", np.median(mean_error), np.median(var_error) )
    print("","Max Error   ", max(abs(mean_error)), max(abs(var_error)) )
    print("","Min Error   ", min(abs(mean_error)), min(abs(var_error)) )
    print()

# Mean
print("len(mean_data): ",len(mean_data))
sub_data = mean_data[mean_data["Algorithm"] == "LSHEP"]
print("len(sub_data): ",len(sub_data))
sub_data = sub_data[sub_data["Dimension"] == 4]
print("len(sub_data): ",len(sub_data))
print("Mean min: ","%.2g"%(min(sub_data["Truth"])))
print("Mean max: ","%.2g"%(max(sub_data["Truth"])))
print("Mean med: ","%.2g"%(np.median(sub_data["Truth"])))
mean_sub_data = sub_data

# Variance
print()
print("len(var_data): ",len(var_data))
sub_data = var_data[var_data["Algorithm"] == "LSHEP"]
print("len(sub_data): ",len(sub_data))
sub_data = sub_data[sub_data["Dimension"] == 4]
print("len(sub_data): ",len(sub_data))
print("Var min: ","%.2g"%(min(sub_data["Truth"])))
print("Var max: ","%.2g"%(max(sub_data["Truth"])))
print("Var med: ","%.2g"%(np.median(sub_data["Truth"])))
var_sub_data = sub_data


# Plot settings
font = dict(family="Times New Roman", size=12)
b = l = 40
r = t = 40
pad = 0
margin = dict(l=l,r=r,b=b,t=t,pad=pad)
kwargs = dict(layout=dict(font=font, titlefont=font, margin=margin,
                          width=700, height=360),
              x_axis_settings=dict(titlefont=font),
              y_axis_settings=dict(titlefont=font),
              config={"displayModeBar": False},
              # x_axis_settings=dict(type="log", autorange=True), 
)

# #      Double histogram of raw values     
# # ========================================
# local_kwargs = kwargs.copy()
# local_kwargs["layout"]["height"] = 320
# p = Plot("", "", "Probability Mass")
# p.add_histogram("", mean_sub_data["Truth"], show_in_legend=False)
# p1 = p.plot(y_range=[0,.2], html=False, **local_kwargs)
# p = Plot("", "I/O Throughput", "Probability Mass")
# p.add_histogram("", var_sub_data["Truth"], show_in_legend=False)
# p2 = p.plot(y_range=[0,.03], html=False, **local_kwargs)
# multiplot([[p1],[p2]], file_name="Raw_Throughput.html", show=True)

# #         Making box plot by training percentage     
# # ======================================================
# print("Collecting Mean_TT_Ratio data..")
# brief_title = "Predicting I/O Throughput Mean"
# error_col = "Relative_Mean_Error"
# training_ranges = [(0,20),(20,40),(40,60),(60,80),(80,100)]
# # Initialize thep lot
# y_axis = "Signed Relative Error in Predicted System Throughput"
# p = Plot(brief_title, "", y_axis)
# for alg in algorithms:
#     alg_data = mean_data[(mean_data["Algorithm"] == alg)]
#     box_values = []
#     box_locations = []
#     for (min_train,max_train) in training_ranges:
#         # Reduce to a local set of data
#         set_data = alg_data[(min_train <= alg_data["Train_Percentage"])]
#         set_data = set_data[(set_data["Train_Percentage"] < max_train)]
#         # Generate the title and axis labels
#         min_val = min(set_data["Train_Percentage"])
#         max_val = max(set_data["Train_Percentage"])
#         x_axis = "[%.0f-%.0f%%] Training"%(min_val,max_val)
#         # Fix the bounds to be mathematically correct
#         if ((min_train, max_train) != training_ranges[-1]):
#             x_axis = x_axis.replace("]",")")
#         box_values += list(set_data[error_col])
#         box_locations += [x_axis]*len(set_data)
#     p.add_box(alg, box_values, box_locations)
# layout = kwargs.get("layout",{})
# layout.update(dict(boxmode="group"))
# local_kwargs = kwargs.copy()
# local_kwargs["layout"] = layout
# print("Generating Mean_TT_Ratio plot..")
# p.plot(y_range=[-10,10], file_name="Mean_TT_Ratio.html", **local_kwargs)
# print("Done")

# # VARIANCE PLOT
# print("Collecting Var_TT_Ratio data..")
# brief_title = "Predicting I/O Throughput Variance"
# error_col = "Relative_Mean_Error"
# training_ranges = [(0,20),(20,40),(40,60),(60,80),(80,100)]
# # Initialize thep lot
# y_axis = "Signed Relative Error in Predicted System Throughput"
# p = Plot(brief_title, "", y_axis)
# for alg in algorithms:
#     alg_data = var_data[(var_data["Algorithm"] == alg)]
#     box_values = []
#     box_locations = []
#     for (min_train,max_train) in training_ranges:
#         # Reduce to a local set of data
#         set_data = alg_data[(min_train <= alg_data["Train_Percentage"])]
#         set_data = set_data[(set_data["Train_Percentage"] < max_train)]
#         # Generate the title and axis labels
#         min_val = min(set_data["Train_Percentage"])
#         max_val = max(set_data["Train_Percentage"])
#         x_axis = "[%.0f-%.0f%%] Training"%(min_val,max_val)
#         # Fix the bounds to be mathematically correct
#         if ((min_train, max_train) != training_ranges[-1]):
#             x_axis = x_axis.replace("]",")")
#         box_values += list(set_data[error_col])
#         box_locations += [x_axis]*len(set_data)
#     p.add_box(alg, box_values, box_locations)
# layout = kwargs.get("layout",{})
# layout.update(dict(boxmode="group"))
# local_kwargs = kwargs.copy()
# local_kwargs["layout"] = layout
# print("Generating Var_TT_Ratio plot..")
# p.plot(y_range=[-80,80], file_name="Var_TT_Ratio.html", **local_kwargs)
# print("Done")

# #      Making Box Plots by Dimension     
# # =======================================
# print("Collecting Mean_Dim data..")
# brief_title = "Predicting I/O Throughput Mean"
# error_col = "Relative_Mean_Error"
# dimensions = [1,2,3,4]
# # Initialize thep lot
# y_axis = "Signed Relative Error in Predicted System Throughput"
# p = Plot(brief_title, "", y_axis)
# for alg in algorithms:
#     alg_data = mean_data[(mean_data["Algorithm"] == alg)]
#     box_values = []
#     box_locations = []
#     for (dim) in dimensions:
#         # Reduce to a local set of data
#         set_data = alg_data[(dim == alg_data["Dimension"])]
#         x_axis = "%i Dimension"%(dim) + ("s" if dim > 1 else "")
#         box_values += list(set_data[error_col])
#         box_locations += [x_axis]*len(set_data)
#     p.add_box(alg, box_values, box_locations)
# layout = kwargs.get("layout",{})
# layout.update(dict(boxmode="group"))
# local_kwargs = kwargs.copy()
# local_kwargs["layout"] = layout
# print("Generating Mean_Dim plot..")
# p.plot(y_range=[-5,5], file_name="Mean_Dim.html", **local_kwargs)
# print("Done")

# VARIANCE PLOT
print("Collecting Var_Dim data..")
brief_title = "Predicting I/O Throughput Var"
error_col = "Relative_Mean_Error"
dimensions = [1,2,3,4]
# Initialize thep lot
y_axis = "Signed Relative Error in Predicted System Throughput"
p = Plot(brief_title, "", y_axis)
for alg in algorithms:
    alg_data = var_data[(var_data["Algorithm"] == alg)]
    box_values = []
    box_locations = []
    for (dim) in dimensions:
        # Reduce to a local set of data
        set_data = alg_data[(dim == alg_data["Dimension"])]
        x_axis = "%i Dimension"%(dim) + ("s" if dim > 1 else "")
        box_values += list(set_data[error_col])
        box_locations += [x_axis]*len(set_data)
    p.add_box(alg, box_values, box_locations)
layout = kwargs.get("layout",{})
layout.update(dict(boxmode="group"))
local_kwargs = kwargs.copy()
local_kwargs["layout"] = layout
print("Generating Var_Dim plot..")
p.plot(y_range=[-5,5], file_name="Var_Dim.html", **local_kwargs)
print("Done")

