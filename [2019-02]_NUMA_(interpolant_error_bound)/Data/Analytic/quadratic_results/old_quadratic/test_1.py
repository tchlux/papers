






dimension = 10

if (test_func_name == "oscillaatory"):
    # Set up the test problem more specifically (after getting its name).
    difficulty = 9.0
    c = np.random.random(size=(dimension,))
    c /= np.sum(c)
    c *= 9.0
    test_function = lambda x: oscillatory(x, c=c)


# Load in the partially collected results if appropriate.
temporary_data_file = "intermediate-analytic.pkl"
if os.path.exists(temporary_data_file):
    d = Data.load(temporary_data_file)
else:
    d = Data(names=["Algorithm", "Function", "SNR", "Train", "Test", "Fit Time", "Predict Time", "Errors"],
             types=[str,         str,        str,   int,     int,    float,      float,          list])


# Run the tests.
for train in train_sizes:
    random.seed(seed)
    np.random.seed(seed)
    print()
    print("="*70)
    print("Generating train points..")
    # Get the Fekete points.
    fekete_file = os.path.join(
        "fekete_points", f"fekete-{train}-{dimension}.pkl")
    try:
        train_points = load(fekete_file)
    except:
        train_points = fekete_points(train, dimension)
        save(train_points, fekete_file)
    print("generating test poitns..")
    test_points = np.random.random(size=(test_size, dimension))
    test_values = test_function(test_points)
    print("  using train points:", train_points.shape)
    print("  using test points: ", test_points.shape)
    # Cycle through different "signal-to-noise ratios"
    for snr in snr_values:
        print()
        print("-"*35)
        # Re-seed the random number generators for each test.
        random.seed(seed)
        np.random.seed(seed)
        snr_string = f"{snr:.2f}"
        print("SNR", snr_string)
        # Evaluate the test function at those points.
        train_values = test_function(train_points)
        # Generate noise (relative to values) and add it on to the values.
        noise_ratio = snr * (np.random.random(size=train_values.shape)*2 - 1)
        train_values += train_values * noise_ratio
        # Print out the time before starting the algorithm execution.
        print()
        print(time.ctime())
        for algorithm in algorithms:
            # Seed each algorithm the same (in case it involved ranndomness).
            random.seed(seed)
            np.random.seed(seed)
            # Initialize the record.
            record = [class_name(algorithm), test_func_name, snr_string, train, test_size]
            print(f"\n {record[0]}")
            # Skip Delaunay for under-determined sets of points.
            if (record[0] in {"DelaunayP1", "ShepMod", "LSHEP"}) and (dimension > train):
                print("  skipping underdetermined setup..")
                continue
            # Skip tests that have already been run and saved.
            elif (record in d[:,:5]):
                print("  skipping already executed test..")
                continue
            # Build the model and test it.
            m = algorithm()
            t = Timer()
            # Fit the model.
            print("  fitting.. ")
            pts = train_points.copy()
            vals = train_values.copy()
            t.start()
            m.fit(pts, vals)
            t.stop()
            print(f"    {t()} seconds")
            record.append( t.total )
            # Evaluate the model.
            print("  predicting.. ")
            pts = test_points.copy()
            t.start()
            guesses = m(pts)
            t.stop()
            print(f"    {t()} seconds")
            record.append( t.total )
            record.append( list(guesses - test_values) )
            # Add this row of values to the recorded data.
            d.append(record)
            # Save the data set into an intermediate file.
            d.save(temporary_data_file)

# Save the final results.
d.save("analytic-results.pkl")
# Remove the intermediate results file.
os.remove(temporary_data_file)
