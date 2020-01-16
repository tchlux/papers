import os
from util.data import Data

# Update the 'fit' data first.
d1 = Data.load("old-fit-final.dill")
d2 = Data.load("fit-intermediate.dill")
# Process the two files into one merged fit data file.
# d3 = d1 + d2
d3 = d1.copy()
d3["Model name"] = (n.replace("SVR","SVR_RBF") for n in d3["Model name"])
# Show progress to user.
print()
print("Saving fit data..")
d3.save("fit-final.dill")
print()

for fname in sorted(os.listdir()):
    if "iozone" not in fname: continue
    if fname[:len("results_")] != "results_": continue
    print()
    print(fname)
    print()
    d1 = Data.load("old-" + fname)
    d2 = Data.load("intermediate_" + fname)
    # Only get the newest data that doesn't describe the problem.
    if "iozone" in fname: d2 = d2[:,-7:]
    else:                 d2 = d2[:,-10:]
    # Pop out the columns that 
    for name in d2.names:
        if name in d1.names:
            print(f"Popping '{name}' from the old data..")
            d1.pop(name)
    d3 = d1 + d2
    print(d3)
    print()
    d3.save(fname)
