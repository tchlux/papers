
# Initially the data didn't have a column for "Training Percentage",
# this code adds that column to the data
# with open("IEEE_Main_Results.csv", "r") as input_file:
#     with open("IEEE_Main_Results_Fixed.csv", "w") as output_file:
#         # Skip the header line
#         names = input_file.readline().strip().split(",")
#         names.insert(6,"Training Percentage")
#         print(",".join(names), file=output_file)
#         sizes = set()
#         count = 0
#         for line in input_file:
#             line = line.strip().split(",")
#             train_size = int(line[6])
#             test_size = int(line[7])
#             train_perc = round((20*train_size / (train_size + test_size)))*5
#             sizes.add(train_perc)
#             line.insert(6,str(train_perc))
#             print(",".join(line), file=output_file)


# Some of the results for delaunay (because it errored) generated
# errors far in excess of 1 (which is impossible). Bound those errors.

with open("IEEE_Main_Results.csv", "r") as input_file:
    with open("IEEE_Main_Results_Fixed.csv", "w") as output_file:
        # Skip the header line
        names = input_file.readline().strip().split(",")
        print(",".join(names), file=output_file)
        for line in input_file:
            line = line.strip().split(",")
            for i in range(11,len(line)):
                val = max(-1,min(1,float(line[i])))
                line[i] = str(val)
            print(",".join(line), file=output_file)
