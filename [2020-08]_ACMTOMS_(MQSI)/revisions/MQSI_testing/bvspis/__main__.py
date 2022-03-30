import os, sys
import numpy as np
from . import spline_fit

module_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

README = f'''
Execute this module by running
  python3 -m {module_name} <data-file> <points-file> [<output-file>]

where "<data-file>" is a path to a file whose contents begin
with an integer N immediately followed by real valued locations
X(1:N) and real function values Y(1:N), and "<points-file>" is
a path to a file starting with an integer M immediately followed
by real valued locations U(1:M) of points to be approximated
with this module over (X,Y).

Values at all U(1:M) are produced by this module and written to 
"<output-file>" if it is provided, otherwise "output.txt".
'''

# Check for the correct number of arguments.
if (len(sys.argv) < 3) or (len(sys.argv) > 4):
    print(README)
    exit()

data_file = sys.argv[1]
points_file = sys.argv[2]
output_file = sys.argv[3] if len(sys.argv) > 3 else "output.txt"

# Read the file containing source data for the approximation.
with open(data_file, "r") as f:
    contents = f.read().strip().split()
    n = int(contents.pop(0))
    x = np.zeros(n, dtype=float)
    y = np.zeros(n, dtype=float)
    for i in range(n):
        x[i] = float(contents.pop(0))
    for i in range(n):
        y[i] = float(contents.pop(0))

# Read the file containing points where approximations are needed.
with open(points_file, "r") as f:
    contents = f.read().strip().split()
    m = int(contents.pop(0))
    u = np.zeros(m, dtype=float)
    for i in range(m):
        u[i] = float(contents.pop(0))

# Construct an approximation to the data.
fit = spline_fit(x, y)

# Create approximations at all new points.
v = fit(u)

# Write the output file.
with open(output_file, "w") as f:
    for i in range(m):
        f.write(f"{v[i]} ")
