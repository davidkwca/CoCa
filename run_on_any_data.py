import sys
sys.path.append("..")

import numpy as np

from models import compare_models
from config import DATA_PATH, CODE_PATH

delimiter = ' '
DZ_inferred = 1

argv = sys.argv
in_file = sys.argv[1]
try:
    out_file = sys.argv[2]
except IndexError:
    out_file = sys.stdout

XY = np.loadtxt(in_file, delimiter=delimiter)
X = XY[:, :-1]
Y = XY[:, -1]

cs_mean, cf_mean = compare_models(X, Y, DZ_inferred)

C = np.abs(cs_mean - cf_mean) / np.max([cs_mean, cf_mean])

conf = 'confounded' if cs_mean < cf_mean else 'causal'

# print(
#     f'This data looks {conf}. The scores are {cs_mean} and {cf_mean} for the causal and confounded models, respectively. This makes a score of C={C}',
# )

with open(out_file, 'a+') as out_f:
    print(in_file, ',', cs_mean, cf_mean, file=out_f)
