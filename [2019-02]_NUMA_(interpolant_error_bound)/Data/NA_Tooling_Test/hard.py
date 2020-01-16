# ==========================================================
#      Test a difficult problem for the box spline code     
# ==========================================================

import numpy as np
from util.system import load, save
from balltree import BallTree
from util.approximate import BoxMesh
from util.math import pairwise_distance

# Load in the example (try for the minimal example, otherwise raw).
try:
    train_x, train_y, test_pt = load("small-dim-bad-example.pkl")
except:
    try:
        train_x, train_y, test_pt = load("small-bad-example.pkl")    
    except:
        train_x, train_y, test_pt = load("bad_box_case.pkl")


# First reduction is in the number of points.
if (train_x.shape[0] > 10):
    print()
    print("train_x: ",train_x.shape)
    print("train_y: ",np.array(train_y).shape)
    print("test_pt: ",test_pt.shape)
    print()

    # Construct a ball tree to get point statistics
    bt = BallTree(train_x)
    d, i = bt.nearest(test_pt, k=20)
    print()
    print("nearest i: ",i)
    print("nearest d: ",d)
    print()
    pd = pairwise_distance(train_x)
    pd[np.diag_indices(len(pd))] = float('inf')
    print(pd)
    print(np.min(pd))
    print()

    # Perform an initial reduction, to a range that still causes the error.
    train_x = train_x[649:975]

    # Now perform a thorough reduction.
    m = BoxMesh()
    while True:
        keep = np.arange(len(train_x))[1:]
        print("remaining pts: ",len(keep)+1, end="\r")
        for i in range(len(keep)):
            m.fit(train_x[keep])
            inds,_ = m(test_pt)
            # Break the search if we know it still doesn't work.
            if (len(inds) == 0):
                train_x = train_x[keep]
                break
            # Otherwise, cycle to drop the next point.
            else: keep[i] = i
        # If we made it through the whole loop and none could be dropped,
        # then break out of the `while` loop.
        else:
            m.fit(train_x[keep])
            inds,_ = m(test_pt)
            # Break the search if we know it still doesn't work.
            if (len(inds) != 0): break
            else: train_x = train_x[keep]
    # Save the smallest possible bad example.
    save((train_x, train_y, test_pt), "small-bad-example.pkl")

# Second reduction is in the number of dimensions.
if (train_x.shape[1]) > 10:
    train_x, train_y, test_pt = load("small-bad-example.pkl")
    # Now perform a thorough reduction.
    m = BoxMesh()
    while True:
        keep = np.arange(train_x.shape[1])[1:]
        print("remaining dims: ",len(keep)+1, end="\r")
        for i in range(len(keep)):
            m.fit(train_x[:,keep])
            inds,_ = m(test_pt[keep])
            # Break the search if we know it still doesn't work.
            if (len(inds) == 0):
                train_x = train_x[:,keep]
                test_pt = test_pt[keep]
                break
            # Otherwise, cycle to drop the next point.
            else: keep[i] = i
        # If we made it through the whole loop and none could be dropped,
        # then break out of the `while` loop.
        else:
            m.fit(train_x[:,keep])
            inds,_ = m(test_pt[keep])
            # Break the search if we know it still doesn't work.
            if (len(inds) != 0): break
            else:
                train_x = train_x[:,keep]
                test_pt = test_pt[keep]
    # Save the smallest possible bad example.
    save((train_x, train_y, test_pt), "small-dim-bad-example.pkl")


# test_pt -= np.min(train_x)
# train_x -= np.min(train_x)
# test_pt /= np.max(train_x)
# train_x /= np.max(train_x)

train_x = np.round(train_x, 2)
test_pt = np.round(test_pt, 2)

print()
print()
print("train_x: ",train_x.shape)
print(train_x)
print()
print(test_pt)
print()
print("fitting..", flush=True)
m = BoxMesh()
m.fit(train_x)
print()
print(m.box_sizes.T)
print()
print("evaluating..", flush=True)
i,w = m(test_pt)
print()
print("i: ",i)
print("w: ",w)
print()

exit()


# ========================================================
#      Test a difficult problem for the Delaunay code     
# ========================================================

from util.math import SMALL
from util.approximate import Delaunay
import numpy as np

print()
print("Loading hard problem..")

a = np.loadtxt("hard_problem.csv", delimiter=",", skiprows=1)
train, test = a[:-1,:], a[-1,:]

train = np.asfortranarray(train.T)
test = np.asfortranarray(test[:,None])

pts_in = train
p_in = test
simp_out = np.ones(shape=(p_in.shape[0]+1, p_in.shape[1]), 
                   dtype=np.int32, order="F")
weights_out = np.ones(shape=(p_in.shape[0]+1, p_in.shape[1]), 
                      dtype=np.float64, order="F")
error_out = np.ones(shape=(p_in.shape[1],), 
                    dtype=np.int32, order="F")

print("Calling delaunay serial subroutine manually..")

m = Delaunay()
m.delaunays(train.shape[0], train.shape[1],
            pts_in, p_in.shape[1], p_in, simp_out,
            weights_out, error_out, extrap=100.0, ibudget=1000,
            eps=2*SMALL)

print("done.")
