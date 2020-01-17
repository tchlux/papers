# Make sure the previous step has been executed.
previous_step = __import__("0-list_data")
directory_file = previous_step.output_file

import os
from util.data import Data
from numpy import percentile

directory_data = Data.load(directory_file)
directory_data.sort()

output_folder = "Reduced"
output_file = "1-data-reduced.csv"

if not os.path.exists(output_file):
    # Decide which percentiles to keep.
    percentiles = [0,10,25,50,75,90,100]
    step_size = 500
    sample = None
    display_skips = False

    print("directory_data: ",directory_data)

    # Function for processing one configuration (row of "directory_data").
    def process_config(config):
        output_file_path = os.path.join(output_folder, os.path.split(config["path"])[-1])
        if os.path.exists(output_file_path):
            if display_skips: print(f"  skipping '{output_file_path}' because it already exists..")
            return
        print()
        print(config["path"])
        # Load the data file.
        print("  loading raw data..")
        config_data = Data.load(config["path"], types=float, sample=sample)
        # Fill all data with the minimum seen so far, reduce to sample steps.
        print("  filling missing values..", flush=True)
        for r,row in enumerate(config_data):
            min_val = row[0]
            for c in range(1,len(row)):
                value = row[c]
                if (value is None):
                    if (not (c-1)%step_size) and all(config_data[v,c] is None for v in range(len(config_data))):
                        break
                    else: row[c] = min_val
                else:
                    min_val = min(min_val, value)
                    row[c] = min_val
        # Reduce the data to the first `step_size` steps, and then every `step_size` steps.
        print("  reducing data..", flush=True)
        steps = list(range(step_size)) + list(range(step_size,config_data.shape[1],step_size))
        old_conf_data = config_data
        config_data = config_data[:,steps].copy()
        del(old_conf_data)
        print("  making summary data..", flush=True)
        # Compute the 0th, 10th, 25th, 50th, 75th, 90th, 100th percentiles.
        summary_data = Data(names=["step"] + [f"{p} percentile" for p in percentiles],
                            types=[int]    + [float]*len(percentiles))
        for column_name in config_data.names:
            # Extract the step number.
            step = int(column_name.split()[-1])
            # Get the percentiles across all the trials, if they exist.
            if config_data[0, column_name] is not None:
                trial_data = list(config_data[column_name])
                values = list(percentile(trial_data, percentiles))
            else: values = [None] * len(percentiles)
            # Store this summary data.
            summary_data.append( [step] + values )
        # Save reduced data file, store reduced path.
        summary_data.save( output_file_path )
        print("  done.", flush=True)


    # from util.parallel import map
    for _ in map(process_config, directory_data): pass

    directory_data["path"] = (
        os.path.join(output_folder, os.path.split(config["path"])[-1])
        for config in directory_data)

    directory_data.save(output_file)
