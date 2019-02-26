import random
import itertools
import argparse
import os
import re

import numpy as np
import pandas as pd
import networkx as nx

import sys
sys.path.append("..")
from models import compare_models
from config import DATA_PATH


def gene_network_parser(fname):
    G = nx.DiGraph()
    with open(fname, 'r') as f:
        for line in f.readlines():
            u, v, e = line.split()
            if int(e):
                G.add_edge(u, v)
    return G


def genetic_network():
    path = f'{DATA_PATH}/gene_network/'
    graph_names = os.path.join(path, 'graph_names.txt')
    traj_names = os.path.join(path, 'trajectory_names.txt')
    with open(graph_names, 'r') as gnames, open(traj_names, 'r') as tnames:
        graph_names = gnames.read().splitlines()
        graph_names = [path + x for x in graph_names]
        traj_names = tnames.read().splitlines()
        traj_names = [path + x for x in traj_names]

    gt = []
    for gname, tname in zip(graph_names, traj_names):
        G = gene_network_parser(gname)
        sample = pd.read_csv(tname, sep='\t')
        sample.drop('Time', axis=1, inplace=True)
        name = re.search('\d+_\w+\d', gname).group()
        gt.append((name, G, sample))

    return gt


def split_tuples(G, sample, r):
    assert r > 1
    nrow, ncol = sample.shape
    causal, confounded = set(), set()
    for i in range(ncol):
        a = sample.columns[i]
        Pa = set(G.pred[a])
        for preds in itertools.combinations(Pa, r - 1):
            if all((Pa.isdisjoint(G.pred[p]) for p in preds)):
                J = tuple([list(sample.columns).index(p) for p in preds])
                causal.add((J, i))
    for i in range(ncol):
        a = sample.columns[i]
        Ca = set((v for u, v in G.edges if u == a))
        for children in itertools.combinations(Ca, r):
            if all(((c1, c2) not in G.edges
                    for c1, c2 in itertools.combinations_with_replacement(
                        children, 2))):
                J = tuple([list(sample.columns).index(c) for c in children])
                confounded.add((J[:-1], J[-1]))
    return causal, confounded


def net_block(G, sample, r, block_number):
    """Returns blocks of 50 pairs which are either directly causally directed, or confounded by a third variable.
    Even samples in the block are directly causal, others are confounded."""
    causal, confounded = split_tuples(G, sample, r)
    k = min(len(causal), len(confounded))
    ij_samples = [0] * (2 * k)
    ij_samples[::2] = random.sample(causal, k)
    ij_samples[1::2] = random.sample(confounded, k)

    # This magically turns the ij_samples list into a list containing li
    ij_blocks = list(itertools.zip_longest(*[iter(ij_samples)] * 50))
    block = filter(None, ij_blocks[block_number])
    return block


def clf_network(gname, G, sample, block, r, block_number):
    result_dir = f'results/synth/gn/'
    os.makedirs(result_dir, exist_ok=True)
    result_fname = f'{result_dir}r{r}_{gname}_{block_number}.csv'
    with open(result_fname, 'w+') as f:
        for q, (i, j) in enumerate(block):
            X = sample.iloc[:, list(i)]
            Y = sample.iloc[:, j].values[:, np.newaxis]

            cs_mean, cf_mean = compare_models(X, Y, 1)
            f.write(','.join([str(q % 2), str(cs_mean), str(cf_mean), '\n']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=0)
    parser.add_argument('-n', type=int, default=0)
    parser.add_argument('-r', type=int, default=2)
    args = parser.parse_args()
    net_number = args.n
    block_number = args.b
    r = args.r

    random.seed(1)
    networks = genetic_network()
    gname, G, sample = networks[net_number]
    try:
        block = net_block(G, sample, r, block_number)
        clf_network(gname, G, sample, block, r, block_number)
    except IndexError:
        print(f'Block {block_number} doesn\'t exist here.')
