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
        
        raw_data = Struct().load(os.path.join(folder, "raw_data.csv"), sep=",")
        nums, info = raw_data.to_numpy_real()
        # Remove the "numeric" columns that are counts of people from normalization
        to_normalize = info.nums[:]
        while any(n >= 45 for n in to_normalize):
            to_normalize.remove(max(to_normalize))
        # Get the normalization numbers used
        shift = np.min(nums[:,to_normalize], axis=0)
        nums[:,to_normalize] -= shift
        scale = np.max(nums[:,to_normalize], axis=0)
        nums[:,to_normalize] /= scale
        # Remove this useless numeric data
        del nums

        # Initialize a holder for all testing data
        all_tests = Struct()

        # Cycle through all folds
        for fold in range(NUM_FOLDS):
            fold += 1
            test_file = os.path.join(folder, f"{fold:02d}-10_test.csv")
            # Load the test data file
            test_data = Struct().load(test_file)
            # Get the names of the predictors and the target
            predictor_names = [test_data.names[i] for i in predictors]
            # Iterate over the algorithms
            for alg in algorithms:
                # Generate an output file and save the data to it
                output_file = os.path.join(folder,"Predictions",f"{alg.__name__}_{fold:02d}-10.csv")
                # Load the prediction output
                prediction_data = Struct().load(output_file, sep=",")
                # Add the prediction output for this algorithm to the test data
                test_data.add_column(prediction_data[:,0], prediction_data.names[0])
            # Concatenate newest test data
            all_tests += test_data

        # Convert the test results back into their categorical form.
        category_tests = Struct(names=raw_data.names)
        for row in all_tests:
            # Get the plain real-valued row and denormalize it
            real_row = np.array(row[:-len(algorithms)])
            real_row[to_normalize] *= scale
            real_row[to_normalize] += shift
            # Convert the real-valued row back into the original row
            raw_row = info.real_to(real_row)
            category_tests.append(raw_row)
        for a in all_tests.names[-len(algorithms):]:
            category_tests.add_column(all_tests[a], a)
        # Save the output of the predictions in one file
        category_tests.save(os.path.join(folder,"all_predictions.csv"))
