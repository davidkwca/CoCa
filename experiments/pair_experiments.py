import glob
import os

import numpy as np
import pandas as pd

import sys
sys.path.append("..")
from models import compare_models
from config import DATA_PATH


def tpair_cf():
    fnames = glob.glob(f'{DATA_PATH}/tubingen-pairs/pair*[0-9].txt')
    for fn in fnames:
        df = pd.read_csv(fn, sep='\s+')
        X = df.iloc[:, 0].values[:, np.newaxis]
        Y = df.iloc[:, 1].values[:, np.newaxis]
        cs_mean, cf_mean = compare_models(X, Y, 1)

        result_dir = 'results/pairs/'
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, 'results.csv'), 'a+') as f:
            f.write(','.join([
                '0',
                str(cs_mean),
                str(cf_mean),
                fn.split('/')[-1].split('.')[0][-4:] + '\n'
            ]))


if __name__ == '__main__':
    tpair_cf()
