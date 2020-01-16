from setup import *

for dt in data_type:
    if "Classification" in dt:
        algorithms = classification_algorithms
    else:
        # Delaunay
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
        
        # Load the raw data and get the transformation function
        raw_data = Struct().load(os.path.join(folder, "raw_data.csv"), sep=",")
        nums, info = raw_data.to_numpy_real()

        # Load the output of all predictions
        results = Struct().load(os.path.join(folder,"all_predictions.csv"), sep=",")
        target = [n for n in results.names if "Recidivism" in n][0]
        predictions = [n for n in results.names if "Prediction" in n]

        if "Classification" in dt:
            _, info = Struct(data=[["Yes"],["No"]]).to_numpy_real()
            truth = np.array([info.to_real([v]) for v in results[target]])
            print()
            print(f"  Yes -> {info.to_real(['Yes'])}")
            print(f"  No --> {info.to_real(['No'])}")
            print()
        else:
            truth = np.array(list(results[target]))
            p = Plot("","Probability of Recidivism","Count",font_family="times")
            p.add_histogram("", truth, show_in_legend=False, num_bins=20)
            p.plot(width=400, height=200, file_name="regression_values_histogram.html")

        guess, _ = results[predictions].to_numpy_real()

        if "Classification" in dt:
            # Get the name of the protected column
            protected = 0
            tgt_idx = results.names.index(target)
            races = sorted(set(v for v in results[results.names[protected]]))
            print()
            for i in range(len(algorithms)):
                print(f"  {algorithms[i]}")
                cm = metric(truth, guess[:,i])
                # Normalize the confusion matrix to be in terms of percentages
                cm = 100 * cm / np.sum(cm)
                # Measure the racial bias in the predicterd recidivism
                count_true = {r:0 for r in races}
                count_wrong = {r:0 for r in races}
                for row in results:
                    label = row[protected]
                    pred_recid = row[-(len(algorithms) - i)]
                    true_recid = info.to_real([row[tgt_idx]])[0]
                    # Update the true count of this label
                    count_true[label] += 1
                    if (round(true_recid) != 1) and (round(pred_recid) == 1):
                        # Count the false positives for this label
                        count_wrong[label] += 1
                # Identify the false positive rate for each race (normalized to percents)
                fp_rate = {}
                for r in races:
                    fp_rate[r] = 100 * count_wrong[r] / count_true[r]
                # Assign the values to the table (for latex friendly printout)
                table_kwargs = {}
                table_kwargs["name"] = algorithms[i].__name__
                table_kwargs["tngn"] = f"{cm[0,0]:.1f}"
                table_kwargs["tngr"] = f"{cm[0,1]:.1f}"
                table_kwargs["trgn"] = f"{cm[1,0]:.1f}"
                table_kwargs["trgr"] = f"{cm[1,1]:.1f}"
                table_kwargs["acc"] = f"{100*np.sum(np.diag(cm)) / np.sum(cm):.1f}"
                table_kwargs["white_nh"] = f'{fp_rate["Black - Non-Hispanic"]:.1f}'
                table_kwargs["black_nh"] = f'{fp_rate["White - Non-Hispanic"]:.1f}'
                table_kwargs["white_h"] = f'{fp_rate["White - Hispanic"]:.1f}'
                table_kwargs["ai_na"] = f'{fp_rate["American Indian or Alaska Native - Non-Hispanic"]:.1f}'
                table_kwargs["asian"] = f'{fp_rate["Asian or Pacific Islander - Non-Hispanic"]:.1f}'
                # Print out the latex table format
                print(table_str.format(**table_kwargs))
            print()
        else:
            # Do processing for regression techniques
            # - get set of pairs (true, guess) for each algorithm
            # - plot set of pairs as points (with dotted y = x line)
            # - np.percentile(0, abs(truth-guess))
            # - np.percentile(25, abs(truth-guess))
            # - np.percentile(50, abs(truth-guess))
            # - np.percentile(75, abs(truth-guess))
            # - np.percentile(100, abs(truth-guess))

            # Get the name of the protected column
            tgt_idx = results.names.index(target)
            # Get the races (sorted for sake of consistency)
            races = sorted(n for n in results.names if "Number " in n)
            # Get the indices of the protected categories
            protected = sorted(
                [c for c in range(len(results.names)) if results.names[c] in races],
                key=lambda c: races.index(results.names[c]))
            # Cycle algorithms
            for i in range(len(algorithms)):
                table_kwargs = {}
                # Collect the signed errors, broken by race, and absolute
                errors = []
                errors_by_race = {r:[] for r in races}
                for row,t,g in zip(results, truth, guess):
                    num_people = sum(row[p] for p in protected)
                    errors += [g[i] - t] * num_people
                    for r,p in zip(races,protected):
                        errors_by_race[r] += [g[i] - t] * row[p]
                # Get the relative absolute errors (scaled 0-100)
                abs_errors = abs(np.array(errors))

                # Generate the plot of errors for this algorithm
                p = Plot("","Guessed Recidivism Probability", "True Recidivism Probability", font_family="times",font_size=20)
                p.add("", [0,1],[0,1], color="rgba(0,0,0,.2)", mode="lines", dash="dot", show_in_legend=False)
                p.add("", guess[:,i], truth, color=p.color(i), shade=False, show_in_legend=False, marker_size=5)
                p.plot(width=500, height=500, x_axis_settings=dict(position=dict(y=-1)),show=False,file_name=f"/Users/thomaslux/Desktop/{algorithms[i].__name__}_regression.html")

                # Get the false positive rates (the median false positive error by race)
                fp_rate = {}
                for r in races:
                    fp_rate[r] = np.percentile([
                        v for v in errors_by_race[r] if v > 0], 50)
                # Generate table keyword arguments
                table_kwargs = {}
                table_kwargs["name"] = algorithms[i].__name__
                table_kwargs["white_nh"] = f'{fp_rate["Number Black - Non-Hispanic"]:.3f}'
                table_kwargs["black_nh"] = f'{fp_rate["Number White - Non-Hispanic"]:.3f}'
                table_kwargs["white_h"] = f'{fp_rate["Number White - Hispanic"]:.3f}'
                table_kwargs["ai_na"] = f'{fp_rate["Number American Indian or Alaska Native - Non-Hispanic"]:.3f}'
                table_kwargs["asian"] = f'{fp_rate["Number Asian or Pacific Islander - Non-Hispanic"]:.3f}'
                table_kwargs["min"] = f'{np.percentile(abs_errors, 0):.3f}'
                table_kwargs["tf"] = f'{np.percentile(abs_errors, 25):.3f}'
                table_kwargs["ff"] = f'{np.percentile(abs_errors, 50):.3f}'
                table_kwargs["sf"] = f'{np.percentile(abs_errors, 75):.3f}'
                table_kwargs["max"] = f'{np.percentile(abs_errors, 100):.3f}'
                table_kwargs["file"] = f"{algorithms[i].__name__}_regression.pdf"
                print(f"  % {algorithms[i].__name__}")
                print(table_str_regr.format(**table_kwargs))



# For the best accuracy biased data:
# - Generate tables confusion matrices
# - Generate plots of (truth vs actual) for regression algorithms
# - Generate bias indicators with each prediction (shift in racial treatment)

# For the interpolation techniques
# - Generate tables of confusion matrices
# - Generate plots of (truth vs actual) for regression algorithms

# Generate debiased data (where the people are weighted according to
# racial discrepancy in represented data)

# For the best accuracy with the debiased (by weighting) data:
# - Generate tables confusion matrices
# - Generate plots of (truth vs actual) for regression algorithms
# - Generate bias indicators with each prediction

# For the interpolant, do the debiasing by weighting at prediction
#  time, present accuracy and bias information


# ========================================
# 
#   Microsoft Surface replacement parts:
# 
#   Service request number
#    1425441109
# 
#   Shipping address
#    807 Cascade Court
#    Blacksburg, Virginia
#    24060
# 
# ========================================
