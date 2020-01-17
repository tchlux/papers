import os
from util.data import Data

RAW_FOLDER = "raw"

# Given a path to a raw data file, it produces the list of strings:
#   [optimizer, dimension, objective, skew, rotation, noise]
def decompose_file_name(name):
    print(name)
    name = name \
           .replace("_(","-")\
           .replace(")_","-")\
           .replace(")","-")\
           .replace("-dimensions","")\
           .replace("-objectified","")\
           .replace("-skewed-","")\
           .replace("-rotated-","")\
           .replace("-noised-","")\
           .replace("mejr","peak")
    return name.split("-")[:-1]

# Initialize a data object for holding all of the configurations.
d = Data(names=["optimizer", "dimension", "objective", "skew", "rotation", "noise", "path"],
         types=[str,         int,         str,         float,  float,      float,   str])

# Read in all of the configurations that have been tested.
for data_dir in os.listdir(RAW_FOLDER):
    if (len(data_dir) > 0) and (data_dir[0] == "."): continue
    for data_file in os.listdir(os.path.join(RAW_FOLDER,data_dir)):
        raw_path = os.path.join(RAW_FOLDER, data_dir, data_file)
        # y = Data.load(raw_path)
        # print(y)
        # exit()
        d.append( decompose_file_name(data_file) + [raw_path] )

# Sort and save the list of configurations available.
d.sort()
print(d)
d.save("data-configurations.csv")

d.summarize()




# with open("output.txt", "w") as f:
#     import pickle

#     with open("ADAGRAD_data.pkl", "rb") as data_f:
#         print("Loading data..")
#         d = pickle.load(data_f)
#         print("  done.")

#     print("Size of data:", (len(d), len(d[0])))
#     print()
#     print("Data names:")
#     print("",d.names)
#     print()
#     for i in range(len(d)):
#         file_name = "_".join(list(map(str,d[i][:-1])))
#         print("", i, file_name)
#     print()
#     print("Everything but last element of first row:")
#     print(d[0][:-1])
#     print()
#     print("Last element of first row: (saved to 'output.txt')")
#     print(d[0][-1], file=f)
