import numpy as np
from util.system import Timer
from parallel import map, builtin_map

def floated(v):
    try: return [float(i) for i in v]
    except: return float(v)

t = Timer()

t.start()
out = builtin_map(floated, list(range(100000000)))
t.stop()
print(np.array(out))
print("Serial time:", t())

t.start()
out = map(floated, list(range(10000)))
t.stop()
print(np.array(out))
print("Parallel time:", t())

