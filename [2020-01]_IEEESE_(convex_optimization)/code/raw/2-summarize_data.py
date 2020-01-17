# Make sure the previous step has been executed.
previous_step = __import__("1-reduce_data")
data_file = previous_step.output_file

import os
from util.data import Data

reduced_data = Data.load(data_file)
reduced_data.sort()

output_file = "2-data-summarized.csv"
output_folder = "Summarized"

if not os.path.exists(output_file):
    summary_data = Data(names=["plot", "series", "path"],
                        types=[str,    str,      str])

    print()
    print("Reduced data:",reduced_data)
    print()

    config_cols = ["dimension", "function", "skew", "rotation", "noise"]

    # Considering each algorithm, average over all values by:
    #   'function'
    #   'skew'
    #   'rotation'
    #   'noise'

    series_by_col = "algorithm"
    plot_by_cols = ["algorithm", "function", "skew", "rotation", "noise"]

    # Cycle each of the differennt plot columns.
    for plot_by_col in plot_by_cols:
        print()
        print('-' * 70)
        # Check for the existence of the output file.
        if (plot_by_col == series_by_col):
            data_file = os.path.join(output_folder, f"{series_by_col}-mean.csv")
        else:
            data_file = os.path.join(output_folder, f"{series_by_col}-mean-by-{plot_by_col}.csv")

        # Store the output path.
        summary_data.append([plot_by_col, series_by_col, data_file])
        # Check to see if this output already exists, if so skip.
        if os.path.exists(data_file):
            print(f"skipping '{data_file}'..")
            continue

        # Identify the different plots that will be created.
        print("plot_by_col: ",plot_by_col)
        print()
        plot_by_values = sorted(set(reduced_data[plot_by_col]))

        # Figure out which configurations to keep (need to be same across all subsets).
        config_cols_to_check = [c for c in config_cols if c not in {plot_by_col, series_by_col}]
        configs_to_keep = reduced_data[config_cols_to_check].unique()
        to_keep = set(range(len(configs_to_keep)))
        for plot_by_value in plot_by_values:
            # Get the data that would be used for this plot, and the configurations.
            plot_data = reduced_data[ reduced_data[plot_by_col] == plot_by_value ]
            plot_configs = plot_data[config_cols_to_check].unique()
            # Remove all configurations that do not appear in this plot's data.
            for i in range(len(configs_to_keep)):
                if i not in to_keep: continue
                if configs_to_keep[i] not in plot_configs: to_keep.remove(i)
        configs_to_keep = configs_to_keep[to_keep]
        # Identify those rows with permitted configurations.
        to_keep = [i for i in range(len(reduced_data))
                   if reduced_data[i, config_cols_to_check] in configs_to_keep]
        twice_reduced_data = reduced_data[to_keep]
        # Print out some information (to show what reduction happened).
        print("configs_to_keep:    ",configs_to_keep.shape)
        print("twice_reduced_data: ",twice_reduced_data.shape)

        # Identify the possible series values that will be used in all plots.
        series_by_values = sorted(set(twice_reduced_data[series_by_col]))

        # Initialize a holder for the "combined data" Data object.
        sample_data = Data.load(twice_reduced_data[0,"path"])
        expected_shape = sample_data.shape
        combined_data = Data(names=[plot_by_col, series_by_col] + list(sample_data.names),
                             types=[str,         str]           + list(sample_data.types))
        mean_indices = [i for i in range(expected_shape[1]) if sample_data.names[i] != "step"]
        print("expected_shape:     ",expected_shape)
        # Make the summarized data files (for easy plotting).
        for plot_by_value in plot_by_values:
            plot_data = twice_reduced_data[twice_reduced_data[plot_by_col] == plot_by_value]
            print()
            print()
            print("plot_by_col:   ",plot_by_col)
            print("plot_by_value: ",plot_by_value)
            print("plot_data:     ",plot_data.shape)
            for series_by_value in series_by_values:
                # Special case for making one plot (by series only).
                if (plot_by_col == series_by_col) and (plot_by_value != series_by_value): continue
                series_data = plot_data[plot_data[series_by_col] == series_by_value]
                print()
                print("  series_by_value: ",series_by_value)
                print("  series_data:     ",series_data.shape)
                # Compute the mean data for all files relevant to this series in this plot.
                data_to_combine = []
                for row in series_data:
                    data_to_combine.append( Data.load(row["path"]) )
                shapes = list(set([d.shape for d in data_to_combine]))
                assert(len(shapes) == 1);
                assert(shapes[0] == expected_shape)
                # Cycle through and meaningfully average aver existing values.
                current_indices = [0] * len(data_to_combine)
                for row_index in range(expected_shape[0]):
                    # Update the "current_indices" to the next row (if no None's). 
                    for i in range(len(data_to_combine)):
                        # If a "None" has already been found, assume they continue.
                        if (current_indices[i] < row_index-1): continue
                        # If there are no "None" values in this row, use it!
                        if (None not in data_to_combine[i][row_index]):
                            current_indices[i] = row_index
                    # Average over the values in each data set.
                    current_step = data_to_combine[0][row_index,"step"]
                    row = [plot_by_value, series_by_value, current_step] + [
                        sum(d[r,c] for (d,r) in zip(data_to_combine, current_indices)) 
                        / len(data_to_combine) for c in mean_indices]
                    combined_data.append(row)
        # Remove a redundant columnn if necessary.
        if (plot_by_col == series_by_col):
            combined_data.pop(combined_data.names[0])

        # Save the combined data.
        combined_data.save(data_file)
        # 
        # plot-series.csv
        #   plot, series, step, [ .. mean percentiles .. ]

    # Save the summary data, to show completion.
    summary_data.save(output_file)
