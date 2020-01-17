# Generate a directory of all of the data files, store in an
# accessible CSV format so that it can be quickly filtered and read.

import os
from util.data import Data

output_file = "0-directory-of-data.csv"
if not os.path.exists(output_file):
    # Given a path to a raw data file, it produces the list of strings:
    #   [optimizer, dimension, objective, skew, rotation, noise]
    def decompose_file_name(name):
        name = name \
               .replace("L_BFGS", "LBFGS")\
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
        alg, dim, func, skew, rotation, noise = name.split("-")[:-1]
        return [alg, int(dim), func, float(skew), float(rotation), float(noise)]



    directories = [f for f in os.listdir() if f.split("-")[-1].isnumeric()]
    algs = sorted({n.split('-')[0] for n in directories})
    dims = sorted({n.split('-')[1] for n in directories})

    data = Data(names=["algorithm", "dimension", "function", "skew", "rotation", "noise", "path"],
                types=[str,         int,         str,        float,  float,      float,   str])

    print("directories: ",directories)
    print("algs: ",algs)
    print("dims: ",dims)
    print()

    for alg_name in algs:
        print("alg_name: ",alg_name)
        for dim_num in dims:
            print("  dim_num: ",dim_num)
            # Get the list of all different test cases.
            dir_name = f"{alg_name}-{dim_num}"
            test_cases = sorted(os.listdir(dir_name))
            for case_name in test_cases:
                print("    case_name: ",case_name)
                path = os.path.join(dir_name, case_name)
                data.append( decompose_file_name(case_name) + [path] )
        print()

    print(data)
    data.save(output_file)
