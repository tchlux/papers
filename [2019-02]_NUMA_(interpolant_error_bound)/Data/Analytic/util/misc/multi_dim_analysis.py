
# This is a small module for creating a "Multi-Dimensional Analysis"
# for some arbitrary data. This will generate a directory of files,
# producing test cases for all combinations of columns of input data
# of varying lengths (starting with all columns, working down to 1).
# For each of those combinations, "NUM_SETS" random unique
# training-testing cases will be generated.

from itertools import combinations
from scipy.spatial import ConvexHull
import random, os

TRAIN_TEST_SPLITS = [((100-i)/100.0,i/100.0) for i in range(5,100,5)]
# ^^ List of the approximate splits to use as ratios of training and
#    testing data for the function 'make_test_data'.
RANDOM_SEED = 0
NUM_SETS = 200
MDA_VERBOSE = True
TEST_IN_HULL = True
TRAIN_WELL_SPACED = True
MAX_REITERATIONS = 3
# Reduce a number to its minimal display form (no unnecessary 0's on right)
CLEAN_NUMBER_STRING = lambda number: str(number).rstrip("0").rstrip(".")

# This class allows for easy traversal of the training-testing files
# inside of an MDA data directory. Simply use it like any other
# iterator (i.e. 'range', 'list', 'dict', etc.), provided with the
# path to the directory upon initialization.
class MDA_Iterator:
    # Given a path to a directory generated by 'make_test_data'
    # function, generate a reasonably ordered list of tuples that hold
    # the absolute paths to pairs of trianing and testing data.
    def __init__(self, top_directory_name):
        self.directory = os.path.abspath(top_directory_name)
        # Store all the file paths in a list
        self.file_paths = []
        # Identify the subdirectory names (<dim>-<col 1>__...)
        subdirs = os.listdir(self.directory)
        subdirs.sort(key=lambda name: -int(name.split("-")[0]))
        # Cycle through all possible combinations of input columns
        for sub in subdirs:
            sub = os.path.join(self.directory, sub)
            sub_name = os.path.basename(sub)
            train_test_dirs = os.listdir(sub)
            # Sort the training and testing directories by ratio
            train_test_dirs.sort(key=lambda name: tuple(map(float,name.split("--")[::-1])))
            # Cycle through all ratios of training and testing split
            for train_test in train_test_dirs:
                train_test = os.path.join(sub, train_test)
                tt_split = os.path.basename(train_test)
                num_files = len(os.listdir(train_test))
                # Cycle through all train-test pairs in order of number
                for num in range(0,num_files,2):
                    num = num // 2
                    train_file = os.path.join(train_test,"Train-%i.csv"%num)
                    test_file = os.path.join(train_test,"Test-%i.csv"%num)
                    if ((not os.path.exists(train_file)) or
                        (not os.path.exists(test_file))):
                        print("Could not find train and test files "+
                              "for number '%i' in '%s'."%(num,train_test))
                    else:
                        self.file_paths.append((train_file, test_file))

    # Tells how many train-test pairs exist in this MDA directory
    def __len__(self):
        return len(self.file_paths)

    # Returns tuples from the list of file names. The 5-tuples are:
    #    (<train file path>, <test file path>, <input dimension>,
    #     <names of each input dimension>, <train-test file number>)
    # 
    # where the types of each of these items are:
    #    ( str, str, int, list(str), int )
    def __iter__(self):
        self.current = 0
        return self

    # The iterator operation for an MDA directory. Returns the next
    # 5-tuples as described by "__iter__".
    def __next__(self):
        if self.current >= len(self.file_paths):
            raise StopIteration
        else:
            train,test = self.file_paths[self.current]
            # First get the file num
            name, num = os.path.split(train)
            # Strip off the train-test split directory name, get the directory
            name = os.path.basename( os.path.split(name)[0] )
            # Now extract the dimension and name from that directory
            dim, name = int(name[:name.index("-")]), name[name.index("-")+1:]
            # Create a list of the column names
            names = name.split("--")
            # Get the number of the test file
            num = int( (num.split(".")[0]).split("-")[1] )
            # Increment and return!
            self.current += 1
            return train, test, dim, names, num


# Function for generating "num_sets" random sets of size "set_size"
# from indices listed in "points". If "well_spaced" then "data" must
# be provided as to measure the distance between points.
def get_rand_sets_of_points(points, set_size, num_sets, data=None,
                            well_spaced=False, needed_points=[]):
    # Pre-compute some necessary values for finding well-spaced points
    if well_spaced:
        # Get all pairs of points and sort them by pairwise distance
        pairs_of_points = list(combinations(points+needed_points,2))
        # Sort by the L2 norm distance between points
        pairs_of_points.sort(
            key = lambda pair: sum((v1 - v2)**2 for v1,v2 in
                zip(data[pair[0]],data[pair[1]]))**(1/2))

    # Generate a set of new points (faster lookup time than list)
    sets_of_points = set()
    for i in range(num_sets*MAX_REITERATIONS):
        if well_spaced:
            # Use the method for generating a well-spaced set of
            # points from the QNSTOP algorithm.
            new_set = set(points)
            # Remove points from the new_set heuristically, starting
            # with the ones that are closest together
            for pair in pairs_of_points:
                # Remove one of the points from the pair randomly if
                # they are both currently in "new_set"
                if all(p in new_set for p in pair):
                    to_remove = random.choice(pair)
                    if to_remove not in needed_points:
                        new_set.remove(to_remove)
                # Break once we have reduced the new set to necessary size
                if len(new_set) == set_size: break
            new_set = tuple(new_set)
        else:
            # If there is no need for ensuring the points are well
            # spaced, then generate a purely random sample
            new_set = random.sample(points,set_size)
        # Check to see if this new set has already been created, if
        # not then add it to the recorded set of points
        if new_set not in sets_of_points:
            sets_of_points.add(new_set)
        if len(sets_of_points) == num_sets: break
    else:
        print(("     WARNING: After %i iterations, could not generate %i unique\n"+
               "              random sets of size %i from %i points.")%(
                   i+1,num_sets,set_size,len(points)))
        print(("              Returning %i sets of unique points instead.")%(
            len(sets_of_points)))
    # Return the sets of points in a sorted order
    return sorted(sets_of_points)


# Function for generating a multi-dimensional analysis test data set
# for analyzing the performance of continuous valued R^d -> R function
# approximation algorithms over varying values of d and varying ratios
# of training and testing data.
# 
# Required Parameters:
#   inputs -- Numpy 'ndarray' with shape (<num points>, <num dims>) of
#             real numbers (dtype=float). These are the points in R^d
#             that will be broken apart into different subsets for
#             testing and training.
#   output -- Numpy 'ndarray' with shape (<num points>,) of real
#             numbers (dtype=float). Must have same length first
#             dimension as "inputs". These will be the response values
#             associated with points in R^d that are to be modelled. 
#             **THESE WILL BE AVERAGED FOR OVERLAPPING INPUT POINTS** 
#             It is important that averaging these values is
#             statistically / mathematically okay.
# 
# Optional Parameters:
#   name         -- A string that can be appended to the standard
#                   parent directory name 'MDA_Data-<rand_seed>'.
#   input_names  -- A list of strings (without commas in them) that
#                   represent the column names of the respectively
#                   indexed input columns. Using this results in more
#                   meaningful naming of sub-directories.  Defaults to
#                   ['Column_1', 'Column_2', ...].
#   dims         -- A list of integers describing the desired size of
#                   input dimension of tests produced for the
#                   MDA. Defaults to be all dimensions ranging from
#                   <num dims> -> 1.
#   num_sets     -- An integer defining the number of random
#                   training-testing sets that should be generated
#                   for each selection of input dimension and
#                   training-testing ratio.
#   rand_seed    -- An integer used to ensure that all of the random
#                   sets of training-testing data are
#                   deterministically produced (for replicability).
#   verbose      -- A boolean that determines whether prints
#                   statements about progress are made.
#   test_in_hull -- A boolean that determines whether or not all
#                   testing points must be inside the convex hull of
#                   the training points. This is particularly useful
#                   if the tests will be used for approximation
#                   algorithms that do not extrapolate. **WARNING**
#                   The use of 'scipy.spatial.ConvexHull' can be slow
#                   to compute in higher than 5 or 6 dimensions.
#   train_well_spaced -- A boolean that determines whether or not
#                        the heuristic for selecting training points
#                        is purely random (leads to quasi-clustering
#                        in high dimensions) or lays preference for
#                        selecting points that are more well-spaced.
def make_test_data(inputs, output, name="", input_names=None,
                   dims=None, num_sets=NUM_SETS,
                   rand_seed=RANDOM_SEED, verbose=MDA_VERBOSE,
                   test_in_hull=TEST_IN_HULL,
                   train_well_spaced=TRAIN_WELL_SPACED):
    # import the "choose" function.
    from util.math import choose

    curr_dir = os.path.abspath(os.path.curdir)
    working_dir = os.path.join(curr_dir, "_".join(
        [name,"MDA_Data",str(rand_seed)]))
    
    if verbose:
        print("Current directory:", curr_dir)
        print("Working directory:", working_dir)
        print()

    # Make directory <name>_MDA_data
    os.makedirs(working_dir, exist_ok=True)
    # Make sure the program performs consistently (reproducibly)
    random.seed(rand_seed)

    # Identify dimension of inputs, initialize dims and input_names
    max_dim = len(inputs[0])
    if type(dims) == type(None):
        dims = list(range(1,max_dim+1))[::-1]
    if type(input_names) == type(None):
        input_names = ["Column_%i"%i for i in range(max_dim)]

    # Cycle through all numbers of input dimensions
    for input_dimension in dims:
        if verbose:
            print()
            print("Input dimension %s:"%input_dimension)

        # Cycle through all possible combinations of input dims
        for in_cols in combinations(range(max_dim),input_dimension):
            names = tuple(input_names[i] for i in in_cols)
            if verbose:
                print()
                print("  "+(", ".join(["%s"]*input_dimension))%names)

            # Filter down the data into the unique input-coordinates,
            # track the response values associated with these unique
            # input coordinates in lists.
            unique_pts = {}
            for row,response in zip(inputs,output):
                in_point = tuple(row[i] for i in in_cols)
                if in_point not in unique_pts:
                    unique_pts[in_point] = [response]
                else:
                    unique_pts[in_point].append(response)
            keys = list(unique_pts.keys())
            keys.sort()

            # Average the values of response for overlapping input points
            run_data = []
            for key in keys:
                response = sum(unique_pts[key]) / len(unique_pts[key])
                run_data.append( key + (response,) )

            # Identify the convex hull points (they cannot be excluded)
            points = list(range(len(run_data)))
            cv_hull = []

            # If testing points should be inside the convex hull, then
            # find the convex hull of the points.
            if test_in_hull:
                # scipy.spatial.ConvexHull can only handle dim > 1.
                if input_dimension > 1:
                    cv_hull = list(ConvexHull(keys).vertices)
                else:
                    cv_hull = [keys.index(min(keys)), keys.index(max(keys))]

                # Remove the convex hull points from the potential test set
                cv_hull.sort()
                for i in cv_hull[::-1]: points.pop(i)
                if verbose:
                    print("    Convex hull contains %i points,"%(len(cv_hull)) +
                          " %i points remaining for testing."%(len(points)))
                    print()

            # Skip sub-data sets that are degenerate
            if len(points) == 0:
                if verbose:
                    print("     WARNING: cannot produce training and testing sets.")
                continue

            # Make the directory for this set of input dimensions
            dim_dir = "%s-"%(len(names)) + "--".join(names)
            os.makedirs(os.path.join(working_dir,dim_dir), exist_ok=True)

            # Cycle through all unique ratios of training / testing
            used_sizes = set()
            for (train,test) in TRAIN_TEST_SPLITS:
                # Adjust train and test size to be true to the data
                train_size = round(len(run_data) * train)
                test_size = len(run_data) - train_size

                # Skip training and testing sizes that have already
                # been used (this happens with small amounts of data)
                if ( ((train_size, test_size) in used_sizes) or
                     (min(train_size,test_size) == 0) or
                     (train_size < len(cv_hull)) or
                     (test_size > len(points)) ):
                    if verbose:
                        print("     WARNING: cannot split %.1f - %.1f."%(
                            100*train, 100*test))
                    continue
                used_sizes.add( (train_size, test_size) )
                # Update the 'true' percentage of training to testing data
                train = 100.0 * train_size / len(run_data)
                test = 100.0 * test_size / len(run_data)
                if verbose:
                    print("    (%0.2f%% - %0.2f%%)  (%i - %i)"%(
                        train, test, train_size, test_size), end="")

                # Make the directory for this split of the data
                split_dir = "%s--%s"%(CLEAN_NUMBER_STRING(round(train,2)),
                                      CLEAN_NUMBER_STRING(round(test,2)))
                os.makedirs(os.path.join(working_dir,dim_dir,split_dir), exist_ok=True)

                # Subtract out the necessary training points from "train_size"
                train_size -= len(cv_hull)

                # Based on how many unique sets can be produced, either
                # use randomness, or cycle all possible combinations.
                if choose(len(points),train_size) <= num_sets:
                    if verbose:
                        print(" using %i (determined) unique combinations."%(
                            choose(len(points),train_size)))
                    sets_of_points = combinations(points,train_size)
                else:
                    if verbose:
                        print(" using %i random combinations."%num_sets)
                    # Pick num_sets random unique selections of training points
                    sets_of_points = get_rand_sets_of_points(
                        points, train_size, num_sets,
                        run_data, train_well_spaced, cv_hull)

                # Cycle through all possible combinations of training points
                for set_num ,train_pts in enumerate(sets_of_points):
                    # Add the convex hull to the training points
                    train_pts = cv_hull + list(train_pts)
                    # The testing points are the remainder (sorted for stability)
                    test_pts = sorted(set(points).difference(train_pts))
                    # Generate the full file paths
                    train_file_name = "Train-%i.csv"%(set_num)
                    test_file_name = "Test-%i.csv"%(set_num)
                    train_file_name = os.path.join(working_dir,dim_dir,
                                                   split_dir,train_file_name)
                    test_file_name = os.path.join(working_dir,dim_dir,
                                                  split_dir,test_file_name)
                    # Skip the creation of files that already exist,
                    # in case someone runs this function again after
                    # having already created the MDA directory but has
                    # increased the number of desired sets
                    if (os.path.exists(train_file_name) and 
                        os.path.exists(test_file_name)): continue
                    # Write training and testing data to individual files
                    with open(train_file_name,"w") as f:
                        for i in train_pts:
                            line = [str(num).rstrip("0").rstrip(".")
                                    for num in run_data[i]]
                            print(",".join(line), file=f)
                    with open(test_file_name,"w") as f:
                        for i in test_pts:
                            line = [str(num).rstrip("0").rstrip(".")
                                    for num in run_data[i]]
                            print(",".join(line), file=f)

        # Add space between different input dimensions
        if verbose: print()

    # Return the path to the working directory to the user
    return working_dir
