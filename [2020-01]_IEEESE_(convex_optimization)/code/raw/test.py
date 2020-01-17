from util.plot import Plot

print("Loading data..")
d = Data.load(os.path.join(dir_name, case_name), types=float, sample=None)
print("Minimizing rows..")
for r,row in enumerate(d):
    min_val = row[0]
    for c in range(1,len(row)):
        if (row[c] is None):
            for c in range(c, len(row)): row[c] = min_val
        else:
            min_val = min(min_val, row[c])
            row[c] = min_val

print("d: ",d)
p = Plot(case_name.replace("_"," "))
skip = 10
for i,row in enumerate(d):
    print(i, end="\r")
    p.add(f"{i}", list(range(0,len(row),skip)), [
        min(1,v) if (v is not None) else v
        for v in row[::skip]], mode="lines")
p.show(y_range=[0,1])


# Reduce all data down to the information at each 100th step.
# 
# Get the min, 10th percentile, quartiles, median, 90th percentile,
#   and max for all trials.
# 
# Plot boxes at each 1000 steps (whiskers are 10th / 90th percentiles,
#   box edges are quartiles, middle line is median). Log scale the
#   x-axis of this plot.



for config in reduced_data[:,config_cols].unique():
    print("Configuration:", config)
    dimension, function, skew, rotation, noise = config
    p = Plot(f"{function} in {dimension} dimensions, skew {skew}, rotation {rotation}, noise {noise}", 
             "step", "objective value", font_family="times")
    for row in reduced_data[reduced_data[:,config_cols] == config]:
        print("",row)
    exit()
    d = Data.load(config["path"])
    print()
    print(config['path'])
    print(d)
    exit()
