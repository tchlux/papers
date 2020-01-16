from util.data import Data
import os, dill

for f in os.listdir("."):
    if ("results_" != f[:len("results_")]): continue
    print(f)
    with open(f, "rb") as f:
        d = dill.load(f)
    print(d)
    d = Data([list(r) for r in d])
    print(d)
    exit()
