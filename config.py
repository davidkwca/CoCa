import os
from os.path import expanduser, join

CODE_PATH = os.path.curdir() # The directory the code is stored in. This is Is only used for the R commands for the comparison plots.
RESULTS_PATH = join(CODE_PATH, 'experiments/results') # This is where the results are stored. Unless you run the code on a server and want to create the plots on your own computer, this should work for you.
TIKZ_PATH = join(CODE_PATH, 'plots/expres') # This is where you want to store the data which can then be used to generate the LaTeX plots.
DATA_PATH = join(CODE_PATH, 'data') # This is where the data is stored.

advi_iter = 30000 # Iterations used for inferring our models.
