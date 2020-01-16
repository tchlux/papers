from util.system import Timer
from util.approximate import Voronoi, Delaunay
from util.data import Data

t = Timer()
print("Loading data..")
d = Data.load("parkinsons_updrs.csv.dill")
print("Converting to matrix..")
num = d.to_matrix()
print("Rescaling data..")
num.data -= num.shift
num.data /= num.scale
print("Extracting matrix..")
x = num.data[:,:-1]
y = num.data[:,-1]

print(x.shape)
print(y.shape)

fit_times = []
approx_times = []
median = 0

for k,(train, test) in enumerate(d.k_fold(only_indices=True)):
    print(f"  {k+1}:10", end="\r", flush=True)
    train_x = x[train]
    train_y = y[train]
    # Time Delaunay "fit" operations
    m = Delaunay()
    m.fit(x,y)
    t.start()
    m(x[0])
    t.stop()
    print("Delaunay 'fit':", t.total)
    # Time the fit.
    model = Voronoi()
    t.start()
    model.fit(train_x, train_y)
    t.stop()
    fit_times.append(t.total)
    # Cycle test and make predictions.
    for j,i in enumerate(test):
        print(f"  {k+1}:10  {j+1}:{len(test)} {median}", end="\r", flush=True)
        t.start()
        model(x[i])
        t.stop()
        approx_times.append(t.total)
        median = sorted(approx_times)[len(approx_times)//2]
    print(" "*70, end="\r")
    print("Fit:    ", sorted(fit_times)[len(fit_times)//2])
    print("Approx: ", sorted(approx_times)[len(approx_times)//2])
    # print("Skipped:", sorted(model.skipped)[len(model.skipped)//2])
    print()
