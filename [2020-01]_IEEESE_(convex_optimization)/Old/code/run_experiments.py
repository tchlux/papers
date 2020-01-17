import os, gzip
import numpy as np
import minimize, objective

STEPS = 50000

# Given a path to a raw data file, it produces the list of strings:
#   [optimizer, dimension, objective, skew, rotation, noise]
def decompose_file_name(name):
    name = name \
           .replace("_(","-")\
           .replace(")_","-")\
           .replace(")","-")\
           .replace("-dimensions","")\
           .replace("-objectified","")\
           .replace("-skewed-","")\
           .replace("-rotated-","")\
           .replace("-noised-","")\
           .replace("_","-")
    # Dimension, Function, Skew, Rotation, Noise
    d, f, s, r, n = name.split("-")[:-1]
    return [f, int(d), float(s), float(r), float(n)]


# Run an experiment
def run_experiment(args, seed=420, trials=100, output="Results",
                   storage="Rotation_Matrices"):
    # Dimension, skew, rotation, noise.
    f, d, s, r, n = args
    # Cycle through objective functions (excluding quadratic and weierstrass)
    obj = objective.functions[f]
    # Load the rotation matrix if it has been computed.
    f = obj.__name__.split("_")[0]
    # Get a rotation matrix file.
    if not os.path.exists(storage): os.makedirs(storage, exist_ok=True)
    rot_matrix_file = f"_{f}_{d}_{s}_{r}_{n}_.txt.gz"
    rot_matrix_file = os.path.join(storage, rot_matrix_file)
    if os.path.exists(rot_matrix_file):
        rotation_matrix = np.loadtxt(rot_matrix_file)
    else: rotation_matrix = None
    # Add on 'skew' to the function.
    print("skewing function..", end="\r")
    func = objective.skew(s, dimension=d)(obj)
    # Add on 'rotation' to the function.
    print("rotating function..", end="\r")
    func = objective.rotate(r, dimension=d, rotation_matrix=rotation_matrix)(func)
    # Save the rotation matrix for later.
    if not os.path.exists(rot_matrix_file):
        np.savetxt(rot_matrix_file, func.rotation_matrix)
    # Add on 'noise' to the function.
    print("noising function..", end="\r")
    func = objective.noise(n, dimension=d)(func)

    # Cycle through minimization algorithms
    # for alg in minimize.algorithms: 
    alg = minimize.L_BFGS

    # Produce an output file name.
    if not os.path.exists(output): os.makedirs(output, exist_ok=True)
    file_name = f"{alg.__name__}_({d}-dimensions)_{func.__name__}.csv"
    file_name = os.path.join(output, file_name)
    if os.path.exists(file_name) or os.path.exists(file_name+".gz"):
        print(" skipping..", file_name)
        return

    print(file_name, flush=True)
    # Collect results of running optimizaiton on this data
    with gzip.open(file_name + ".gz", "wt") as output_file:
        random_generator = np.random.RandomState(seed)
        # Print the header to the file.
        print(",".join([f"Step {i}" for i in range(STEPS+1)]),file=output_file)
        for t in range(trials):
            start = func.rand(d, random_generator)
            obj_vals = list(map(str,map(
                func, alg(func, func.grad, start, budget=STEPS))))
            print(",".join(obj_vals) +
                  ","*(len(obj_vals)-STEPS-1),
                  file=output_file)

# --------------------------------------------------------------------

if __name__ == "__main__":
    # SPLITTER = '''# --------------------------------------------------------------------'''
    # TO_APPEND = '''run_experiment((%s, %s, %s, %s, %s))'''
    # TARGET_DIR = os.path.join(os.path.expanduser("~"), "Git",
    #                           "distributed", "Jobs")
    # # Make the target directory.
    # os.makedirs(TARGET_DIR, exist_ok=True)
    # # Read the contents of this file.
    # with open(__file__, "r") as f: my_contents = f.read().split(SPLITTER)[0]

    funcs = [1, 2, 3, 4] # <- subquad, supquad, saddle, multimin
    dims  = [10, 100, 1000]
    skew  = [0., .5,  .99]
    rot   = [0., .5,  1.]
    noise = [0., .5,  1.]
    jobs = []

    # Cycle through the jobs.
    from itertools import product
    for i,v in enumerate(product(funcs, dims, skew ,rot, noise)):
        # Skip combinations of skew, rotation, and noise
        _, _, s, r, n = v
        if (s != 0) and (r != 0): continue
        if (s != 0) and (n != 0): continue
        if (r != 0) and (n != 0): continue

        # Add to a jobs list (to run all jobs in parallel).
        jobs.append(v)

        # # Show what's being generated.
        # # print(my_contents + TO_APPEND%tuple(map(str,v)))
        # print(v)
        # with open(os.path.join(TARGET_DIR, f"job-{i}.py"), "w") as f:
        #     print(my_contents + TO_APPEND%tuple(map(str,v)), file=f)

    # Run the experiments in parallel.
    from util.parallel import map as pmap
    for _ in pmap(run_experiment, jobs, chunk=False): pass
