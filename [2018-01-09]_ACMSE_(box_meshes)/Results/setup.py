import numpy as np
import fmodpy
import time

# Update the link arguments to include blas and lapack
module_link_args = ["-lblas", "-llapack", "-lgfortran"]
meshes = fmodpy.fimport("meshes.f90", requested_funcs=[
    "length", "most_central_point", "linear_basis_value", "train_vm",
    "predict_vm", "train_mbm", "train_ibm", "predict_box_mesh"]
                        ,module_link_args=module_link_args)
                        # ,working_directory="fmodpy_meshes"
                        # ,force_rebuild=False
                        # ,verbose=True)

ERROR_TOLERANCE = 0.0

def run_tests():
    # length
    vector = np.array([1,2,3,4,5,6], dtype=np.float64)
    print("length")
    print(" default value")
    print("  meshes.length(vector): ",meshes.length(vector))
    for norm in range(1,4):
        true_length = np.sum(vector**norm) ** (1 / norm)
        print(" norm: ", norm)
        print("  meshes.length(vector, norm): ",meshes.length(vector, norm))
        print("  True length with 'norm':     ",true_length)
    print()

    # most_central_point
    print("most_central_point")
    points = np.random.random((25,4))
    center = (np.max(points, axis=0) + np.min(points, axis=0)) / 2
    distances = np.sum((points - center)**2, axis=1)
    true_center = np.argmin(distances)
    calc_center = meshes.most_central_point(points.T) - 1
    print("  true_center:                         ",  true_center)
    print("  meshes.most_central_point(points.T): ",  calc_center)
    print()

    # linear_basis_value
    print("linear_basis_value")
    for v in np.linspace(-1.4,1.4,7):
        print(" v:", v)
        print("  max(abs(1 - v), 0):           ",max(1 - abs(v), 0))
        print("  meshes.linear_basis_value(v): ",  meshes.linear_basis_value(v))

def test_time(model, n=100, d=10):
    x = np.random.random(size=(n,d))
    y = np.random.random(size=(n,))
    start = time.time()
    model.fit(x,y)
    fit_time = time.time() - start
    start = time.time()
    model(x[:1,:])
    eval_time = time.time() - start
    return fit_time, eval_time

def make_plot(surf, random=False, append=False):
    from util.plotly import Plot
    mult = 5
    fun = lambda x: np.cos(x[0]*mult) + np.sin(x[1]*mult)
    np.random.seed(0)
    p = Plot()
    low = -0.1
    upp = 1.1
    dim = 2
    plot_points = 2000
    N = 30
    if random:
        x = np.random.random(size=(N,dim))
    else:
        N = int(round(N ** (1/dim)))
        x = np.array([r.flatten() for r in np.meshgrid(np.linspace(low,upp,N), np.linspace(low,upp,N))]).T
    y = np.array([fun(v) for v in x])
    # x = np.array([[0.5,0.5], [0.2,0.2], [0.2,0.8], [0.8,0.8], [0.8,0.2]])
    # y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    p.add("Training Points", *x.T, y)
    surf.fit(x,y)
    p.add_func("VMesh", surf, *([(low,upp)]*dim), plot_points=plot_points)
    p.plot(file_name="vmesh.html", append=append)

class Mesh:
    def __init__(self, error_tolerance=ERROR_TOLERANCE):
        self.error_tolerance = error_tolerance
    # Wrapper for 'predict' that returns a single value for a single
    # prediction, or an array of values for an array of predictions
    def __call__(self, x:np.ndarray, *args, **kwargs):
        single_response = len(x.shape) == 1
        if single_response:
            x = np.array([x])
        if len(x.shape) != 2:
            raise(Exception("ERROR: Bad input shape."))
        response = np.asarray(self.predict(x, *args, **kwargs), dtype=float)
        # Return the response values
        return response[0] if single_response else response
    # Functions to be overwritten by subclasses
    def fit(): pass
    def predict(): pass

# ====================================
#      Class for the Voronoi Mesh     
# ====================================

class VoronoiMesh(Mesh):
    # Fit a set of points
    def fit(self, points, values):
        self.points = np.asarray(points.T.copy(), order="F")
        self.values = np.asarray(values.copy(), order="F")
        self.control_points = np.ones(self.points.shape, dtype=np.float64, order="F")
        self.control_values = np.ones(self.values.shape, dtype=np.float64, order="F")
        self.coefficients = np.ones(self.values.shape, dtype=np.float64, order="F")
        self.control_dots = np.ones((points.shape[0],)*2, dtype=np.float64, order="F")
        num_control_points = -1
        # Calculate the parameters for the VM
        _, _, _, _, num_control_points = (
            meshes.train_vm(self.points, self.values, self.control_points,
                            self.control_values, self.coefficients,
                            self.control_dots, num_control_points,
                            self.error_tolerance))
        self.control_points = self.control_points[:,:num_control_points]
        self.control_values = self.control_values[:num_control_points]
        self.coefficients = self.coefficients[:num_control_points]
        self.control_dots = self.control_dots[:num_control_points,:num_control_points]
        self.control_dots = np.asarray(self.control_dots, order="F")

    # Generate a prediction for a new point
    def predict(self, xs):
        to_predict = np.asarray(xs.T, order="F")
        predictions = np.ones((xs.shape[0],), dtype=np.float64, order="F")
        meshes.predict_vm(self.control_points, self.control_values,
                          self.coefficients, self.control_dots,
                          to_predict, predictions)
        return predictions

# ====================================
#      Class for the Max Box Mesh     
# ====================================

class MaxBoxMesh(Mesh):
    # Fit a set of points
    def fit(self, points, values):
        self.points = np.asarray(points.T.copy(), order="F")
        self.values = np.asarray(values.copy(), order="F")
        self.control_points = np.ones(self.points.shape, dtype=np.float64, order="F")
        self.control_values = np.ones(self.values.shape, dtype=np.float64, order="F")
        self.box_sizes = np.ones((self.points.shape[0]*2,self.points.shape[1]),
                                 dtype=np.float64, order="F")
        self.coefficients = np.ones(self.values.shape, dtype=np.float64, order="F")
        num_control_points = -1
        # Calculate the parameters for the VM
        _, _, _, _, num_control_points = (
            meshes.train_mbm(self.points, self.values, self.control_points,
                             self.control_values, self.box_sizes,
                             self.coefficients, num_control_points,
                             self.error_tolerance))
        self.control_points = self.control_points[:,:num_control_points]
        self.control_values = self.control_values[:num_control_points]
        self.box_sizes = self.box_sizes[:,:num_control_points]
        self.coefficients = self.coefficients[:num_control_points]

    # Generate a prediction for a new point
    def predict(self, xs):
        to_predict = np.asarray(xs.T, order="F")
        predictions = np.ones((xs.shape[0],), dtype=np.float64, order="F")
        meshes.predict_box_mesh(self.control_points, self.box_sizes,
                                to_predict, self.coefficients,
                                predictions)
        return predictions

# ==========================================
#      Class for the Iterative Box Mesh     
# ==========================================

class IterativeBoxMesh(Mesh):
    # Fit a set of points
    def fit(self, points, values):
        self.points = np.asarray(points.T.copy(), order="F")
        self.values = np.asarray(values.copy(), order="F")
        self.control_points = np.ones(self.points.shape, dtype=np.float64, order="F")
        self.control_values = np.ones(self.values.shape, dtype=np.float64, order="F")
        self.box_sizes = np.ones((self.points.shape[0]*2,self.points.shape[1]),
                                 dtype=np.float64, order="F")
        self.coefficients = np.ones(self.values.shape, dtype=np.float64, order="F")
        num_control_points = -1
        # Calculate the parameters for the VM
        _, _, _, _, num_control_points = (
            meshes.train_ibm(self.points, self.values, self.control_points,
                             self.control_values, self.box_sizes,
                             self.coefficients, num_control_points,
                             self.error_tolerance))
        self.control_points = self.control_points[:,:num_control_points]
        self.control_values = self.control_values[:num_control_points]
        self.box_sizes = self.box_sizes[:,:num_control_points]
        self.coefficients = self.coefficients[:num_control_points]

    # Generate a prediction for a new point
    def predict(self, xs):
        to_predict = np.asarray(xs.T, order="F")
        predictions = np.ones((xs.shape[0],), dtype=np.float64, order="F")        
        meshes.predict_box_mesh(self.control_points, self.box_sizes,
                                to_predict, self.coefficients,
                                predictions)
        return predictions


if __name__ == "__main__":
    # run_tests()
    # algs = (VoronoiMesh, MaxBoxMesh, IterativeBoxMesh)
    algs = (MaxBoxMesh,) # IterativeBoxMesh VoronoiMesh
    t = 0.0
    for a in algs:
        surf = a(t)
        n = 60
        d = 6400
        for d in range(800, 6401, 800):
            print("Testing %i points with %i attributes..."%(n,d),end="\r")
            times = test_time(surf, n, d)
            print("",a.__name__, n, d, times)
        exit()

        # for n in range(100,501,200):
        #     for d in range(2,23,10):
        # make_plot(surf, append=(a!=algs[0]))




# PERFORMANCE ON RANDOM POINTS

# MaxBoxMesh 100 2 (0.022318124771118164, 7.295608520507812e-05)
# MaxBoxMesh 100 12 (0.07199287414550781, 0.00011110305786132812)
# MaxBoxMesh 100 22 (0.1020669937133789, 7.605552673339844e-05)
# MaxBoxMesh 300 2 (0.6656620502471924, 6.914138793945312e-05)
# MaxBoxMesh 300 12 (1.9727509021759033, 0.00010085105895996094)
# MaxBoxMesh 300 22 (2.861636161804199, 0.00011610984802246094)
# MaxBoxMesh 500 2 (3.35809326171875, 6.723403930664062e-05)
# MaxBoxMesh 500 12 (9.792442798614502, 0.00012421607971191406)
# MaxBoxMesh 500 22 (13.866275072097778, 0.0001499652862548828)
# IterativeBoxMesh 100 2 (0.0209500789642334, 5.602836608886719e-05)
# IterativeBoxMesh 100 12 (0.06354498863220215, 6.413459777832031e-05)
# IterativeBoxMesh 100 22 (0.09317994117736816, 7.295608520507812e-05)
# IterativeBoxMesh 300 2 (0.6882739067077637, 6.794929504394531e-05)
# IterativeBoxMesh 300 12 (1.8802838325500488, 0.00010013580322265625)
# IterativeBoxMesh 300 22 (2.7759220600128174, 0.00010800361633300781)
# IterativeBoxMesh 500 2 (3.5089383125305176, 0.00010609626770019531)
# IterativeBoxMesh 500 12 (9.386315107345581, 0.00011897087097167969)
# IterativeBoxMesh 500 22 (13.358751058578491, 0.0001537799835205078)
# VoronoiMesh 100 2 (0.04131889343261719, 8.082389831542969e-05)
# VoronoiMesh 100 12 (0.07755208015441895, 7.772445678710938e-05)
# VoronoiMesh 100 22 (0.08210206031799316, 9.703636169433594e-05)
# VoronoiMesh 300 2 (1.8706669807434082, 0.0001239776611328125)
# VoronoiMesh 300 12 (5.332704067230225, 0.00023984909057617188)
# VoronoiMesh 300 22 (7.378243923187256, 0.0003311634063720703)
# VoronoiMesh 500 2 (9.870266914367676, 0.0001552104949951172)
# VoronoiMesh 500 12 (36.77297902107239, 0.0006740093231201172)
# VoronoiMesh 500 22 (58.44468903541565, 0.0010089874267578125)



# VoronoiMesh 100 1 (0.028046131134033203, 8.821487426757812e-05)
# VoronoiMesh 600 1 (13.291136026382446, 0.00013184547424316406)
# VoronoiMesh 1100 1 (143.2126660346985, 0.00034499168395996094)

# MaxBoxMesh 100 1 (0.013724803924560547, 0.0005772113800048828)
# MaxBoxMesh 600 1 (4.753708124160767, 6.29425048828125e-05)
# MaxBoxMesh 1100 1 (34.72099423408508, 6.604194641113281e-05)
# MaxBoxMesh 1600 1 (126.38264513015747, 8.416175842285156e-05)



# BAD PERFORMANCE BEFORE REDUCING REDUNDANCY

# VoronoiMesh 100 2 (0.4004678726196289, 0.00018978118896484375)
# VoronoiMesh 100 12 (1.3829350471496582, 0.0004489421844482422)
# VoronoiMesh 100 22 (2.4140052795410156, 0.00075531005859375)
# VoronoiMesh 300 2 (30.187440156936646, 0.0010960102081298828)
# VoronoiMesh 300 12 (110.48724412918091, 0.0036149024963378906)
# VoronoiMesh 300 22 (192.25026607513428, 0.006303071975708008)
# VoronoiMesh 500 2 (233.96421098709106, 0.002859830856323242)
# VoronoiMesh 500 12 (1611.1297039985657, 0.009881019592285156)
