import numpy as np
from util.data import read_struct
from setup import MaxBoxMesh, IterativeBoxMesh, VoronoiMesh
import time

MONTH_MAP = {
    'jan':1,
    'feb':2,
    'mar':3,
    'apr':4, 
    'may':5,
    'jun':6,
    'jul':7,
    'aug':8,
    'sep':9,
    'oct':10,
    'nov':11,
    'dec':12,
}

DAY_MAP = {
    'sun':1,
    'mon':2,
    'tue':3,
    'wed':4,
    'thu':5,
    'fri':6,
    'sat':7,
}

if __name__ == "__main__":
    print()

    #                        Data Preparation                       
    # ==============================================================
    data_parkinsons = read_struct("parkinsons_updrs.csv")
    data_parkinsons = data_parkinsons[:(len(data_parkinsons)//50) * 4]
    data_forestfire = read_struct("forestfires.csv")
    data_hpcio = read_struct("hpc_file_io.csv")
    data_hpcio = data_hpcio[:(len(data_hpcio)//2)]
    print("Data Header Information:")
    print("",len(data_parkinsons.dtype.names), data_parkinsons.dtype.names[5:])
    print("",len(data_forestfire.dtype.names), data_forestfire.dtype.names)
    print("",len(data_hpcio.dtype.names), data_hpcio.dtype.names)
    print()

    # Organize (and normalize) the parkinsons data
    data_parkinsons_x = np.vstack(tuple(data_parkinsons[h] for h in 
                                        data_parkinsons.dtype.names[6:])).T
    data_parkinsons_y = data_parkinsons[data_parkinsons.dtype.names[5]]
    min_parkinsons_x = np.min(data_parkinsons_x, axis=0)
    max_parkinsons_x = np.max(data_parkinsons_x, axis=0)
    data_parkinsons_x = (data_parkinsons_x - min_parkinsons_x) / (
        max_parkinsons_x - min_parkinsons_x)

    # Organize (and normalize) the forest fire data
    months = np.array([MONTH_MAP[v] for v in data_forestfire["month"]])
    days   = np.array([DAY_MAP[v]   for v in data_forestfire["day"]])
    names = list(data_forestfire.dtype.names[:-1])
    names.remove("month"); names.remove("day")
    data_forestfire_x = np.vstack((months,days,)+tuple(data_forestfire[h] for h in names)).T
    data_forestfire_y = data_forestfire[data_forestfire.dtype.names[-1]]+1
    min_forestfire_x = np.min(data_forestfire_x, axis=0)
    max_forestfire_x = np.max(data_forestfire_x, axis=0)
    data_forestfire_x = (data_forestfire_x - min_forestfire_x) / (
        max_forestfire_x - min_forestfire_x)

    # Organize (and normalize) the hpc io data
    data_hpcio_x = np.vstack(tuple(data_hpcio[h] for h in 
                                        data_hpcio.dtype.names[:-1])).T
    data_hpcio_y = data_hpcio[data_hpcio.dtype.names[-1]]
    min_hpcio_x = np.min(data_hpcio_x, axis=0)
    max_hpcio_x = np.max(data_hpcio_x, axis=0)
    data_hpcio_x = (data_hpcio_x - min_hpcio_x) / (
        max_hpcio_x - min_hpcio_x)
    

    print("Processed Data Sizes:")
    print("",data_parkinsons_x.shape, data_parkinsons_y.shape)
    print("",data_forestfire_x.shape, data_forestfire_y.shape)
    print("",data_hpcio_x.shape, data_hpcio_y.shape)
    print()

    #                      Algorithm Evaluation                     
    # ==============================================================

    RANDOM_SEED = 0
    TRAIN_TEST_SPLIT = (80,20)
    VALIDATION_FOLDS = 10
    ERROR_TOLERANCE_RANGE = np.linspace(0,2,11)


    # print("Data,Tolerance,Train_Size,Test_Size,Fold_Num,Model_Name,Fit_Time,Eval_Time,Min_Error,Max_Error,Median_Error,Average_Error")
    # for (data_name,x,y) in [
    #         ("Parkinsons", data_parkinsons_x, data_parkinsons_y)
    #         ,("Forest Fire", data_forestfire_x, data_forestfire_y)
    #         ,("HPC I/O", data_hpcio_x, data_hpcio_y)]:

    #     np.random.seed(RANDOM_SEED)
    #     indices = np.arange(len(y))
    #     train_size = int(round(len(y) * (TRAIN_TEST_SPLIT[0]/100)))

    #     for t in ERROR_TOLERANCE_RANGE:
    #         for fold in range(VALIDATION_FOLDS):
    #             np.random.shuffle(indices)
    #             train_indices = indices[:train_size]
    #             test_indices = indices[train_size:]
    #             train_x = np.asarray(x[train_indices], dtype=np.float64)
    #             train_y = np.asarray(y[train_indices], dtype=np.float64)
    #             test_x = np.asarray(x[test_indices], dtype=np.float64)
    #             test_y = np.asarray(y[test_indices], dtype=np.float64)
    #             for model_name, model in [
    #                     ("Max Box Mesh", MaxBoxMesh(t))
    #                     ,("Iterative Box Mesh", IterativeBoxMesh(t))
    #                     ,("Voronoi Mesh", VoronoiMesh(t))]:
    #                 start = time.time()
    #                 model.fit(train_x, train_y)
    #                 fit_time = time.time() - start
    #                 start = time.time()
    #                 approx_y = model(test_x)
    #                 eval_time = time.time() - start
    #                 error = (approx_y - test_y) / test_y
    #                 print(data_name, t, len(train_y), len(test_y),
    #                       fold, model_name, fit_time, eval_time,
    #                       min(error), max(error), np.median(error),
    #                       np.mean(error), sep=",")

    performance = read_struct("ACM_results.csv")
    tols = sorted(np.unique(performance["Tolerance"]))
    # print(performance.dtype)
    
    # # Store the best tolerance values for each model
    # best_models = {
    #     "Max Box Mesh":       [1.2, 1.8, 0.6],
    #     "Iterative Box Mesh": [0.4, 1.8, 1.8],
    #     "Voronoi Mesh":       [0.2, 1.0, 2.0]
    # }

    for i,(data_name,x,y) in enumerate([
            ("HPC I/O", data_hpcio_x, data_hpcio_y)            
            ,("Forest Fire", data_forestfire_x, data_forestfire_y)
            ,("Parkinsons", data_parkinsons_x, data_parkinsons_y)]):
        # Initialize the indices and randomness necessary for validation
        train_size = int(round(len(y) * (TRAIN_TEST_SPLIT[0]/100)))
        # Extract a reduced set of performance data (for determining
        # best error tolerance when building a model)
        perf_data = performance[performance["Data"]==data_name]
        for model_name, template in [
                ("Max Box Mesh", MaxBoxMesh)
                ,("Iterative Box Mesh", IterativeBoxMesh)
                ,("Voronoi Mesh", VoronoiMesh)]:
            np.random.seed(RANDOM_SEED)
            indices = np.arange(len(y))
            perf_model = perf_data[perf_data["Model_Name"]==model_name]
            # Identify the optimal error tolerance based on previous data
            tol_performances = []
            for t in tols:
                perf_tol = perf_model[abs(perf_model["Tolerance"]-t)<.01]["Average_Error"]
                tol_performances.append(np.mean(abs(perf_tol)))
            t = tols[np.argmin(tol_performances)]
            errors = []
            for fold in range(VALIDATION_FOLDS):
                np.random.shuffle(indices)
                train_indices = indices[:train_size]
                test_indices = indices[train_size:]
                train_x = np.asarray(x[train_indices], dtype=np.float64)
                train_y = np.asarray(y[train_indices], dtype=np.float64)
                test_x = np.asarray(x[test_indices], dtype=np.float64)
                test_y = np.asarray(y[test_indices], dtype=np.float64)
                model = template(t)
                model.fit(train_x, train_y)
                approx_y = model(test_x)
                error = (approx_y - test_y) / test_y
                errors += list(error)
            print(data_name, model_name, t, min(tol_performances),len(errors),*errors,sep=",")

#  22 ('total_UPDRS', 'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP', 'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE')
#  13 ('X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area')
#  5 ('File_Size', 'Record_Size', 'Threads', 'Frequency', 'Mean_Throughput')

# Processed Data Sizes:
#  (468, 16) (468,)
#  (517, 12) (517,)
#  (432, 4) (432,)

# [('Data', '<U11'), ('Tolerance', '<f8'), ('Train_Size', '<i8'), ('Test_Size', '<i8'), ('Fold_Num', '<i8'), ('Model_Name', '<U18'), ('Fit_Time', '<f8'), ('Eval_Time', '<f8'), ('Min_Error', '<f8'), ('Max_Error', '<f8'), ('Median_Error', '<f8'), ('Average_Error', '<f8')]
