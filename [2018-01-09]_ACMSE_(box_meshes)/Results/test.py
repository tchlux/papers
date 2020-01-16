import numpy as np

pts = np.array([[0.50, 0.50],
                [0.20, 0.20],
                [0.20, 0.80],
                [0.80, 0.80],
                [0.80, 0.20]])
con = np.vstack((pts[0],pts[4]))


mat = np.matmul(pts, con.T)
print("Points:\n",pts)
print("Control Points:\n",con)
print("Point Dots:\n",mat)
