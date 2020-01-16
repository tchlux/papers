from util.approximate import Voronoi
from util.data import Data
from util.system import save, load

try:
    lengths = load()
except:
    d = Data.load("raw_data.csv")
    lengths = []
    for i in range(1,11):
        print("i:",i)
        train = Data.load(f"{i:02d}-10_train.csv", sample=None)
        test = Data.load(f"{i:02d}-10_test.csv", sample=None)
        # Get the column ranges for inputs and outputs.
        in_idxs = list(range(len(train.names)))[:train.names.index("Recidivism Likelihood")]
        out_idxs = list(range(len(train.names)))[train.names.index("Recidivism Likelihood"):]
        # in_cols = train.names[:train.names.index("Recidivism Likelihood")]
        # out_cols = train.names[train.names.index("Recidivism Likelihood"):]
        # Get the matrices.
        train_mat = train.to_matrix()
        test_mat = test.to_matrix()
        train_x = train_mat[:,in_idxs]
        test_x = test_mat[:,out_idxs]
        m = Voronoi()
        m.fit(train_x)
        lengths += [len(ids) for ids,wts in m(test_x)]
        save(lengths)

print("len(lengths): ",len(lengths))

from util.plot import Plot
from util.stats import cdf_fit

p = Plot("Distribution of Number of Influencers",
         "Number of records used to make prediction",
         "CDF value")
cdf = cdf_fit(lengths)
p.add_func("lengths", cdf, cdf(), show_in_legend=False, color=p.color(1))
p.show()

exit()

from util.approximate import Voronoi, NearestNeighbor, NeuralNetwork, DecisionTree
from util.approximate import ShepMod, BoxMesh, LSHEP, Delaunay
from util.approximate.testing import test_plot

# model = NearestNeighbor()
# model = Voronoi()
# model = DecisionTree()
# model = NeuralNetwork()

# model = BoxMesh()
# model = LSHEP()
# model = ShepMod()
model = Delaunay()

p, x, y = test_plot(model, random=True, N=40)
p.title = str(model)
p.show()

exit()



