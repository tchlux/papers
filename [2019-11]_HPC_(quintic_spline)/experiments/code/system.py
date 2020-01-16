# Save "data" in a file titled "file_name" using pickle.
def save(data, file_name="_save.pkl"):
    import dill as pickle
    with open(file_name, "wb") as f:
        pickle.dump(data, f)


# Load data from a pickle file titled "file_name".
def load(file_name="_save.pkl"):
    import dill as pickle
    with open(file_name, "rb") as f:
        data = pickle.load(f)
    return data
