
import os
import numpy as np
from util.plotly import Plot
from util.data import Struct
from util.parallel import map as parallel_map

# Get an approximation algorithm to make predictions

# Get util algorithms for regression
from util.approximate import Voronoi
from util.approximate import NearestNeighbor
from util.approximate import Delaunay
# Get sklearn algorithms for classification and regression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

# Get an evaluation metric and print out its result (confusion matrix)
from sklearn.metrics import confusion_matrix as metric


NUM_FOLDS = 10

main_folder = "Processed_Data"
data_type = ["Full_Classification", "Condensed_Regression"][::-1]
age_type = ["Age_Cat", "Age_Num"]

# Algorithms for classification and regression
classification_algorithms = [NearestNeighbor, DecisionTreeClassifier, MLPClassifier]
regression_algorithms = [Voronoi, NearestNeighbor, DecisionTreeRegressor, MLPRegressor]
# Delaunay

# The predictor columns
class_cat_preds = list(range(53))[4:] # 5 races went to 4 categories
class_num_preds = list(range(50))[4:] # 5 races went to 4 categories
regrs_cat_preds = list(range(54))[:-5] # 5 races went to 5 columns
regrs_num_preds = list(range(51))[:-5] # 5 races went to 5 columns


# name     -- name of the algorithm
# tngn     -- true no recidivism, guessed no recidivism
# tngr     -- true no recidivism, guessed recidivism
# trgn     -- true recidivism, guessed no recidivism
# trgr     -- true recidivism, guessed recidivism
# acc      -- accuracy
# white_nh -- white non-hispanic change
# black_nh -- black non-hispanic change
# white_h  -- white hispanic change
# ai_na    -- american indian, native alaskan change
# asian    -- asian or pacific islander change
table_str = '''
  \\begin{{tabular}}{{c c}}
    \\vspace{{3mm}} \\\\
    \\multicolumn{{2}}{{c}}{{\\textit{{{name}}}}} \\\\
    \\vspace{{-2mm}} \\\\
    \\begin{{tabular}}{{c c c}}
      & GN & GR \\\\ \\cline{{2-3}}
      TN & \\multicolumn{{1}}{{|c|}}{{{tngn}\\%}} & \\multicolumn{{1}}{{|c|}}{{{tngr}\\%}} \\\\ \\cline{{2-3}}
      TR & \\multicolumn{{1}}{{|c|}}{{{trgn}\\%}} & \\multicolumn{{1}}{{|c|}}{{{trgr}\\%}} \\\\ \\cline{{2-3}}
       & & \\vspace{{-2mm}} \\\\
      \\multicolumn{{3}}{{r}}{{Accuracy: {acc}\\%}} \\\\
    \\end{{tabular}} & 
    \\begin{{tabular}}{{|c l|}}
      \\hline
      {white_nh}\\% & White - NH \\\\
      {black_nh}\\% & Black - NH \\\\
      {white_h}\\% & White - H \\\\
      {ai_na}\\% & AI or NA \\\\
      {asian}\\% & Asian or PI \\\\
      \\hline
    \\end{{tabular}} \\\\
  \\end{{tabular}}
'''

# name     -- name of the algorithm
# file     -- plot file name
# acc      -- accuracy
# white_nh -- white non-hispanic median absolute error
# black_nh -- black non-hispanic median absolute error
# white_h  -- white hispanic median absolute error
# ai_na    -- american indian, native alaskan median absolute error
# asian    -- asian or pacific islander median absolute error
# min      -- minimum absolute error
# tf       -- twenty-fifth percentile absolute error
# ff       -- fifty-fifth percentile absolute error
# sf       -- seventy-fifth percentile absolute error
# max      -- max absolute error
table_str_regr = '''
  \\begin{{tabular}}{{c c}}
    \\vspace{{3mm}} \\\\
    \\multicolumn{{2}}{{c}}{{\\textit{{{name}}}}} \\\\
    \\multicolumn{{2}}{{c}}{{ \\includegraphics[width=2in,height=2in]{{{file}}}}} \\vspace{{3mm}} \\\\
    \\begin{{tabular}}{{c|c}}
        Min & {min} \\\\
        $25^{{th}}$ & {tf} \\\\
        $50^{{th}}$ & {ff} \\\\
        $75^{{th}}$ & {sf} \\\\
        Max & {max} \\\\
    \\end{{tabular}} &
    \\begin{{tabular}}{{|c l|}}
      \\hline
      {white_nh} & White - NH \\\\
      {black_nh} & Black - NH \\\\
      {white_h} & White - H \\\\
      {ai_na} & AI or NA \\\\
      {asian} & Asian or PI \\\\
      \\hline
    \\end{{tabular}} \\vspace{{3mm}} \\\\
  \\end{{tabular}}
'''
