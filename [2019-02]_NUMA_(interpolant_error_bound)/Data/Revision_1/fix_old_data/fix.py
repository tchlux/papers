import os, dill

# Update the system path to only point to the local directory (no imports from elsewhere).
import sys
to_remove = [p for p in sys.path if "fix_old_data" in p]
if len(to_remove) > 0:
    for p in to_remove: sys.path.pop(sys.path.index(p))
# Import the master util Data object.
from util.data import Data
print()
print(sys.path)
print(Data)
# Now delete the already-imported module so it doesn't interfere with ".util"
_ = sys.modules.pop("util")
del _
_ = sys.modules.pop("util.data")
del _
# Now reset the path so the local util is used for the import.
sys.path = to_remove[:1] + sys.path


# Update the 'fit-final.dill' file.
f = "fit-final.dill"
with open(f, "rb")  as raw: d = dill.load(raw)
nd = Data(d, names=d.names, types=d.types)
nd.save("old-" + f)

# Update all results files.
for f in sorted(os.listdir()):
    if "results_" != f[:len("results_")]: continue
    print()
    print(f)
    with open(f, "rb") as raw: d = dill.load(raw)
    print()
    print(d.names)
    print(d.types)
    print()
    nd = Data(d, names=d.names, types=d.types)
    nd.save("old-" + f)
    print("Updated..")

