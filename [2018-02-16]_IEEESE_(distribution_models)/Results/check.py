import os, pickle
import numpy as np
from util.data import read_struct

# Frequency,File Size,Record Size,Num Threads,Test,Algorithm,Training
# Percentage,Train Size,Test Size,Fit Time,Eval Time,KS
# Statistic,Average Absolute Error,Median Error,Average Error 

data = read_struct("IEEE_Main_Results.csv", verbose=True)
# for h in list(data.dtype.names)[]
# print(low_throughput.shape)

print("Result lines:        ", len(data))
subdata = data[data["Test"] == "readers"]
print("'readers' lines:     ", len(subdata))
subdata = subdata[subdata["Algorithm"] == "Delaunay"]
print("'Delaunay' lines:    ", len(subdata))
indices = sorted((i for i in range(len(subdata))), key=lambda i:
                 data[i]["KS Statistic"])
median_error = subdata[indices[len(indices)//2]]
print("Median KS Stat point:", median_error)
exit()

# subdata = data[data["Training Percentage"] == 95]
# subdata = data[data["Test"] == ]
ks_values = [.1568, .1879, .2251, .3110]

for alg in sorted(set(subdata["Algorithm"])):
    print(alg)
    perf_data = subdata[subdata["Algorithm"] == alg]["KS Statistic"]
    for ks in ks_values:
        print("","%.4f"%(ks),"%.4f"%(sum(np.where(perf_data > ks,1,0)) / len(perf_data)))


# names = list(data.dtype.names)
# print(names)
# for n in names[11:]:
#     col = data[n]
#     print()
#     print(n)
#     print("Min: ",min(col))
#     print("Max: ",max(col))
#     print("Bad: ", sum(np.where(abs(col) > 1, 1, 0)))
#     indices = np.arange(len(col))
#     print("Algs:",sorted(set(data["Algorithm"][indices[abs(col) > 1]])))

# from util.plotly import Plot
# p = Plot()
# for alg in sorted(set(data["Algorithm"])):
#     print("Finding '%s'"%alg)
#     subdata = data[data["Algorithm"] == alg]["KS Statistic"]
#     print("Adding...")
#     p.add_histogram(alg, subdata)
# p.plot()

