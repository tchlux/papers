import os
from util.data import Data

# --------------------------------------------------------------------

# STEP 1
sample = None
prefix = '[0]-' if sample is None else '[0]-sample-'
test = "readers" # "initial-writers" # "readers"

# The conditions on taking each of the steps.
step_1 = not os.path.exists(prefix+"mu-reduced.csv.gz")
step_2 = not os.path.exists(prefix+test+".csv.gz")
step_3 = not os.path.exists(prefix+test+".csv.pkl")

print()
print("Step 1..")
if step_1:
    print(" loading 'mu.csv.gz' and reducing..")
    data = Data.load("mu.csv.gz", sample=sample)
    to_keep = ['', 'host', 'media', 'freq', 'fsize', 'rsize', 'threads', 'iter', 'subiter', 'test', 'throughput']
    data = data[to_keep]
    data.save(prefix+"mu-reduced.csv.gz")
elif (step_2 or step_3):
    print(" loading 'mu-reduced.csv.gz'..")
    data = Data.load(prefix+"mu-reduced.csv.gz", sample=sample)
print(" done.")

# STEP 2
print()
print("Step 2..")
if step_2:
    print(f" reducing to '{test}' test..")
    full_data = data
    # Pick which columns to remove.
    to_remove = ['', 'host', 'media',  'iter', 'subiter', 'test']
    to_keep = [n for n in data.names if n not in to_remove]
    # reduce to the specific test, and throw out the columns I don't want
    data = data[data["test"] == test][to_keep].copy()
    del(full_data)
    print(data)
    # Stack up the throughput values.
    data.stack('throughput')
    data.inflate('throughput')
    print(data)
    data.save(prefix+test+".csv.gz")
elif step_3:
    data = Data.load(prefix+test+".csv.gz", verbose=False, sample=sample)
print(" done.")

# STEP 3
print()
print("Step 3..")
if step_3:
    print(f" saving a '.pkl' version for fast loading..")
    data.save(prefix+test+".pkl")
print(" done.")

# --------------------------------------------------------------------
