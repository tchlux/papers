from setup import *

for dt in data_type:
    if "Classification" in dt:
        algorithms = classification_algorithms
    else:
        algorithms = regression_algorithms

    for at in age_type:
        print(dt, at)
        #      Get the predictor and target columns     
        # ==============================================
        if "Classification" in dt:
            if "Cat" in at:
                predictors = class_cat_preds
            else:
                predictors = class_num_preds
        else:
            if "Cat" in at:
                predictors = regrs_cat_preds
            else:
                predictors = regrs_num_preds

        #      Cycle through predicting for each algorithm     
        # =====================================================
        folder = os.path.join(main_folder, dt, at)
        # Make an output folder
        os.makedirs(os.path.join(folder, "Predictions"), exist_ok=True)
        for fold in range(NUM_FOLDS):
            fold += 1
            train_file = os.path.join(folder, f"{fold:02d}-10_train.csv")
            test_file = os.path.join(folder, f"{fold:02d}-10_test.csv")
            # Load the train and test data files
            print(f"Loading {train_file}...")
            train_data = Struct().load(train_file)
            print(f"Loading {test_file}...")
            test_data = Struct().load(test_file)
            # Get the names of the predictors and the target
            predictor_names = [train_data.names[i] for i in predictors]
            print("Converting to real...")
            # Get the "train" and "test" as numpy real matrices
            train, train_info = train_data[predictor_names].to_numpy_real()
            test, test_info = test_data[predictor_names].to_numpy_real()
            # Iterate over the algorithms
            for alg in algorithms:
                # Construct and fit the model
                print(f"  Fitting {alg.__name__} model..")
                model = alg()
                model.fit(train[:,:-1], train[:,-1])
                print(f"  Evaluating {alg.__name__} model...")
                # Evaluate the model at all points
                def predict(row):
                    if (type(model) != Delaunay):
                        row = row.reshape((1, len(row)))
                        prediction = model.predict(row).reshape((1,))
                    else:
                        print()
                        print("(/",", ".join(list(map(str,row))),"/)")
                        pts, wts = model.points_and_weights(row)
                        print()
                        print(pts)
                        print(wts)
                        prediction = [sum(test[pts,-1] * wts)]+list(pts)+list(wts)
                        print(prediction)
                        exit()
                    return prediction[0]
                # Generate all predictions and decide which type of
                # map to use (for efficiency) based on the algorithm.
                if (alg == Delaunay):
                    predictions = np.array([v for v in map(predict, test[:,:-1])])
                    output_names = [f"{alg.__name__} Predictions"]+[]
                    # TODO: Need to add columns for points and weights
                else:
                    predictions = np.array([v for v in map(predict, test[:,:-1])])
                    output_names = [f"{alg.__name__} Predictions"]
                    predictions = predictions.reshape((len(predictions),1))
                # Generate an output file and save the data to it
                output_file = os.path.join(folder, "Predictions", f"{alg.__name__}_{fold:02d}-10.csv")
                prediction_data = Struct(data=predictions, names=output_names)
                print(f"  Saving {output_file}...")
                prediction_data.save(output_file)
                print()


