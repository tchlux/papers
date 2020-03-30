import os
from util.data import Data

temp_results = "[1]-temporary-results.csv"

results_file = "[2]-results.csv.gz"
if not os.path.exists(results_file):
    raw = Data.load(temp_results, sample=None)
    raw.sort()
    raw.save(results_file)

raw_rank_results = "[2]-raw_rank_probabilities.csv.gz"
processed_results = "[2]-processed-results.csv.gz"


# These plots are not used in the final paper.
GENERATE_RANK_PROBABILITY_DATA = False
MAKE_RANK_PROBABILITY_PLOT = False

# --------------------------------------------------------------------
if GENERATE_RANK_PROBABILITY_DATA:
    res = Data.load(results_file, sample=None)
    print(res)
    from util.stats.rank import rank_probability
    algs = sorted(set(res["fit"]))
    samples = sorted(set(res["sample"]))
    ranks = Data(names=["sample", "fit", "rank 0", "rank last"],
                 types=[int,      str,   float,    float])
    print("samples: ",samples)
    for s in samples:
        print("s: ",s)
        d = res[res["sample"] == s]
        for a in algs:
            print("  a: ",a)
            # Get all of the values for this algorithm (in one list).
            mine = d[d["fit"] == a]
            my_values = [v for row in mine for v in row[3:]]
            # Get all of the values for each other algorithm (in a list per).
            all_other_values = []
            for b in algs:
                if (b == a): continue
                others = d[d["fit"] == b]
                other_values = [v for row in others for v in row[3:]]
                if len(other_values) > 0:
                    all_other_values.append(other_values)
            # Get the rank 0 probability for "me".
            ranks.append([s, a, 
                          rank_probability(my_values, all_other_values,rank=0),
                          rank_probability(my_values, all_other_values,rank=-1)
            ])
            ranks.save(raw_rank_results)
    exit()
    # Probability of having lowest absolute error at each sample size.

# --------------------------------------------------------------------
if MAKE_RANK_PROBABILITY_PLOT:
    # ranks_last = Data.load("raw_rank_last_probabilities.csv")
    ranks = Data.load(raw_rank_results)
    print(ranks)
    from util.plot import Plot
    samples = sorted(set(ranks["sample"]))
    algs = sorted(set(ranks["fit"]))
    p = Plot("ranked first", "probability", "sample size")
    for a in algs:
        d = ranks[ranks["fit"] == a]
        x = list(d["sample"])
        y = list(d["rank 0"])
        p.add(a, x, y, mode="markers+lines")
    p.show()

    p = Plot("ranked last", "probability", "sample size")
    for a in algs:
        d = ranks[ranks["fit"] == a]
        x = list(d["sample"])
        y = list(d["rank last"])
        p.add(a, x, y, mode="markers+lines")
    p.show()
    exit()


# Probability of having lowest absolute error at each percentile for:
#    sample size 20
#    sample size 80
#    sample size 200

res = Data.load(results_file, sample=None)

data = res[:,:3].copy()
# Compute the KS statistic
data[f"KS"] = (max(map(abs,row[3:])) for row in res)
data[f"1-norm"] = (sum(map(abs,row[3:])) for row in res)
data[f"2-norm"] = (sum(v**2 for v in row[3:])**(1/2) for row in res)
print(data)

# from numpy import percentile
# percentiles = [0,25,50,75,100]
# for p in percentiles:
#     print(f"Computing {p} percentile..")
#     data[f"er {p}"] = map(lambda row: percentile(row[3:], p), res)
# for p in percentiles:
#     print(f"Computing {p} absolute percentile..")
#     data[f"abs er {p}"] = map(lambda row: percentile([abs(v) for v in row[3:]], p), res)
# print(data)

data.save(processed_results)
# --------------------------------------------------------------------

