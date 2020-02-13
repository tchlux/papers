# Get the name of a class as listed in python source file.
def class_name(obj):
    if type(obj) == type: return (repr(obj)[1:-2].split(".")[-1])
    else:                 return (repr(obj)[1:-2].split(".")[-1]).split(" ")[0]

# Import the base classes for approximation and the wrappers for approximation.
from util.approximate.base import Approximator, WeightedApproximator
from util.approximate.wrappers import unique, condition

# Import all defined approximation algorithms.

# Weighted approximators (define predictions as convex weighted sums).
from util.approximate.delaunay import Delaunay, DelaunayP1, qHullDelaunay
# Regression techniques.
from util.approximate.neural_network import NeuralNetwork

