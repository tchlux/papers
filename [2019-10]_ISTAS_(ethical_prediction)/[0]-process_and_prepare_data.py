import os
from util.data import Struct, read_struct
import numpy as np

SEED = 0
NUM_FOLDS = 10

# Read in the cleaned recidivism data.
data = Struct()
cleaned_data_file = "recidivism_cleaned.pkl.gz"

if not os.path.exists(cleaned_data_file):
    data.load("3-Year_Recidivism_for_Offenders_Released_from_Prison.csv")
    print()
    data.summarize(26)
    print()
    #      Identify rows that should be removed     
    # ==============================================
    # 
    # Get the unique values in each column, remove rows with values
    # that are under-supported across the data (< 1% of data) for
    # those columns that will be used primarily in the study
    # 
    to_remove = set()
    predictors = data.names[:10+1]
    # Retrieve the dictionary of dictionaries of counts
    counts = data.unique()
    # Process the data looking for undersupported values
    for n in predictors:
        for v in sorted(counts[n], key=lambda v: -counts[n][v]):
            support = (100 * counts[n][v] / len(data))
            if (counts[n][v] < 100):
                bad_rows = list(data[n] == v)
                print(f"# Removing rows with value '{v}' in column '{n}' because they only have {support:.2f}% support ({counts[n][v]} instances):")
                print(f"#   {bad_rows}")
                print("#")
                to_remove.update(set(bad_rows))
    # Reduce down to the set of points to be kept
    to_keep = [i for i in range(len(data)) if i not in to_remove]
    data = data[to_keep]
    # Summarize the newly reduced data
    print()
    data.summarize(26)
    print()
    # Save the reduced data for convenience
    print("Saving the data...")
    data.save(cleaned_data_file)
    data.save("recidivism_cleaned.csv")
else:
    data = data.load(cleaned_data_file)


# Assign the missing values to be their own category
data[data["Release Type"] == None,8] = "N/A"
data[data["Main Supervising District"] == None,9] = "N/A"

# No offense subtypes are reused under different types (except for
# "other violent") which exists under "Other" and under "Violent". 
# Remove the offense type and classification because it provides redundant information

# Get the names of the columns that will be used for prediction
predictor_names = data.names[:10+1]
# Remove "Convicting Offense Type"
predictor_names.pop(6)
# Remove "Recidivism Reporting Year"
predictor_names.pop(1)
# Get the protected column (race)
protected_column = predictor_names.pop(1)
# Get the "Recidivism" target from the data
target_column = predictor_names.pop(-1)

# ============================================
#      Generate all of the processed data     
# ============================================

for CONVERT_AGE_TO_NUMERIC in {True, False}:
    for USE_UNIQUE_CRIMES in {True, False}:

        print(f"Numeric Age:   {CONVERT_AGE_TO_NUMERIC}")
        print(f"Unique Crimes: {USE_UNIQUE_CRIMES}")
        print()

        # Reduce to just the crime data we want to build predictions off of
        crimes = data[[protected_column] + predictor_names + [target_column]]

        # If age should be numeric, generate some values.
        if CONVERT_AGE_TO_NUMERIC:
            # =============================================
            #      Generate a Map for Transforming Age     
            # =============================================

            # Reassign the "Age At Release" to be a numeric column using range mean,
            # 18 is used as minimum age, average lifespan in US is maximum age.
            age_at_release_map = {
                "Under 25":(18+25)/2,
                "25-34":(25+34)/2,
                "35-44":(35+44)/2,
                "45-54":(45+54)/2,
                "55 and Older":(55+78.74)/2,
            }
            new_age_at_release = [age_at_release_map[v] for v in crimes["Age At Release"]]
            crimes["Age At Release"] = new_age_at_release

        # =============================================================
        #      Generate the Set of Unique Crimes for Interpolation     
        # =============================================================

        if USE_UNIQUE_CRIMES:
            # Reduce the data to the set of unique crimes that have been committed
            # according to our "predictor" data.
            unique_crimes = {}
            unique_races = set()
            for row in crimes:
                crime = row[:-1]
                race = crime.pop(0) # Pop out the race
                crime = tuple(crime)
                unique_crimes[crime] = unique_crimes.get(crime,[]) + [(row[-1],race)]
                # Add all of the "race" values
                unique_races.add(race)

            # Store the set of unique races as a sorted list
            unique_races = sorted(unique_races)

            # Convert the unique crimes into a concise listing of unique crimes
            for crime in unique_crimes:
                results = unique_crimes[crime]
                # Count the number of recidivism cases there were
                perc_positive = sum(v[0] == "Yes" for v in unique_crimes[crime]) / len(results)
                # Count the number of each race that were present
                num_per_race = tuple(sum(v[1] == r for v in results) for r in unique_races)
                # Update the record to be the concise listing (of all same length)
                unique_crimes[crime] = (perc_positive,) + num_per_race

            # This struct only contains unique crimes with recidivism
            # probabilities instead of all of the source data.
            unique_crimes = Struct(
                data = (crime+unique_crimes[crime] for crime in unique_crimes),
                names = predictor_names + ["Recidivism Likelihood"] + [
                    f"Number {r}" for r in unique_races]
            )
            # Save this data (for later verification)
            unique_crimes.save("recidivism_unique_crime_probability.csv")
            crimes = unique_crimes


        # ================================
        #      Generate the Path Name     
        # ================================

        folder = "Processed_Data"
        # Pick which folder to go into based on type of data
        if USE_UNIQUE_CRIMES:
            folder = os.path.join(folder, "Condensed_Regression")
        else:
            folder = os.path.join(folder, "Full_Classification")

        # Make the directory structure
        os.makedirs(folder, exist_ok=True)

        # Pick which sub-folder to go into based on type of "age" column
        if CONVERT_AGE_TO_NUMERIC:
            folder = os.path.join(folder, "Age_Num")
        else:
            folder = os.path.join(folder, "Age_Cat")

        # Make the directory structure
        os.makedirs(folder, exist_ok=True)

        # Save the processed data (for this type of problem)
        output_file = os.path.join(folder, f"raw_data.csv")
        crimes.save(output_file)

        # =======================================
        #      Generate the real-valued data     
        # =======================================

        # Save the numeric translation of the data (ready for splitting)
        nums, info = crimes.to_numpy_real()

        to_normalize = info.nums[:]
        # Remove the "numeric" columns that are counts of people from normalization
        while any(n >= 45 for n in to_normalize):
            to_normalize.remove(max(to_normalize))

        print(f"Normalizing {[n for i,n in enumerate(info.names) if i in to_normalize]}")
        # Normalize the numeric data (that is going to be used for prediction)
        shift = np.min(nums[:,to_normalize], axis=0)
        nums[:,to_normalize] -= shift
        scale = np.max(nums[:,to_normalize], axis=0)
        nums[:,to_normalize] /= scale
        crimes = Struct(data=nums, names=info.names)
        output_file = os.path.join(folder, "numeric_data.csv")
        crimes.save(output_file)
        print(f"Shift: {shift}")
        print(f"Scale: {scale}")
        print()

        # =========================================
        #      Generate Folded Train-Test Data     
        # =========================================

        # Generate the train / test pairs using k-folds of the data.
        for i,(train,test) in enumerate(crimes.k_fold(k=NUM_FOLDS, seed=SEED)):
            train_file = os.path.join(folder, f"{i+1:02d}-{NUM_FOLDS}_train.csv")
            test_file = os.path.join(folder, f"{i+1:02d}-{NUM_FOLDS}_test.csv")
            train.save(train_file)
            test.save(test_file)

