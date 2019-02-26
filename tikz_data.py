import pandas as pd
import numpy as np
import glob
import itertools

import os

from sklearn.metrics import auc

from config import RESULTS_PATH, TIKZ_PATH

# def data_to_tikz(fnames):
#     if len(fnames) == 1:
#         fname = fnames[0]
#     try:
#         dx = re.search('dx(\d+)', fname).group()
#         dz = re.search('dz(\d+)', fname).group()
#     except:
#         dx = 0
#         dz = 0
#     colnames = ('confounded', 'cs_mean', 'cf_mean', 'cfc_mean')
#     df = pd.concat(
#         (pd.read_csv(f, names=colnames, header=None) for f in fnames),
#         ignore_index=True)
#     df.confounded = df.confounded.map({
#         'both': 2,
#         'confounded': 1,
#         'causal': 0,
#         1: 1,
#         0: 0
#     })
#     print(df.head(5))
#     correct = (((df.confounded == 1) &
#                 (np.max(df[['cs_mean']], axis=1) < df.cf_mean)) |
#                ((df.confounded == 0) &
#                 (np.max(df[['cf_mean']], axis=1) < df.cs_mean)))
#     df['correct'] = correct
#     df['confidence'] = np.abs(df.cf_mean - df.cs_mean) / np.max(
#         np.abs(df[['cf_mean', 'cs_mean']]), axis=1)
#     df = df.sort_values(by=['confidence'], ascending=False)
#     df = df.reset_index()
#     return df, dx, dz

# def tikz(fnames):
#     df, dx, dz = data_to_tikz(fnames)
#     df['dr'] = np.cumsum(df.correct) / np.cumsum(np.ones(len(df.correct)))
#     df.index += 1
#     with open(f'{TIKZ_PATH}/synth_{dx}_{dz}.dat', 'w+') as f:
#         df.dr.to_csv(f, index=True, sep=' ')

# def pair_tikz(i=1):
#     df = pd.read_csv(
#         f'/home/dk/contact/bayes/results/tpairs/{i}_results.csv', header=None)
#     df.columns = ('cf', 'cs_mean', 'cf_mean', 'pair')
#     df['conf'] = (df.cf_mean - df.cs_mean) / np.max(
#         np.abs(df[['cf_mean', 'cs_mean']]), axis=1)
#     weights = pd.read_csv(
#         'data/tubingen-pairs/pairmeta.txt', header=None, sep='\s+')
#     weights = weights.iloc[:, -1]
#     df['weights'] = weights
#     df = df.sort_values(by=['pair'], ascending=True)
#     df = df.reset_index()
#     cf = pd.read_csv(f'data/tubingen-pairs/confounded.txt', header=None)
#     cf = cf.iloc[:, 0]
#     df['cf'] = cf
#     df = df[df.cf >= 0]
#     df['correct'] = df['cf'] == (df['conf'] >= 0)
#     df.conf = np.abs(df.conf)
#     df = df.sort_values(by=['conf'], ascending=False)
#     df = df.reset_index()
#     df.index += 1
#     df['dr'] = np.cumsum(df.correct * df.weights) / np.cumsum(df.weights)
#     with open(f'{TIKZ_PATH}/pairs{i}.dat', 'w+') as f:
#         df.dr.to_csv(f, index=True, sep=' ')

# def gn_tikz():
#     fnames = glob.glob('/home/dk/contact/bayes/results/gene_net_multi3/*.csv')
#     tikz(fnames)

# def synth_tikz(date=None):
#     if not date:
#         date = datetime.date.today().strftime('%m-%d')
#     fnames = glob.glob(f'/home/dk/contact/bayes/results/{date}/*.csv')
#     for fname in fnames:
#         tikz([fname])

# def load_df(fname, columns=('confounded', 'cs_mean', 'cf_mean')):
#     df = pd.read_csv(fname, header=None)
#     df = df.loc[:, :2]
#     df.columns = columns
#     try:
#         df.confounded = df.confounded.map({
#             'both': 2,
#             'confounded': 1,
#             'causal': 0
#         })
#     except TypeError:
#         pass
#     return df


def gather_data(dx=1, dz=5, ndata='*', addition='*', data='dr'):
    fnames = glob.glob(
        '{}/synth/{}/**/{}_n{}_dx{}_dz{}.csv'.format(RESULTS_PATH, data,
                                                     addition, ndata, dx, dz),
        recursive=True)
    fnames_cf = filter(lambda x: '/cf/' in x, fnames)
    fnames_cs = filter(lambda x: '/cs/' in x, fnames)

    df_cf = pd.concat(
        [pd.read_csv(fn, header=None) for fn in fnames_cf]).iloc[:, 1:]
    df_cs = pd.concat(
        [pd.read_csv(fn, header=None) for fn in fnames_cs]).iloc[:, 1:]
    df_cf.columns = df_cs.columns = [
        'cs_mean', 'cf_mean', 'spectral_beta', 'indep_beta', 'indep_pval'
    ]
    df_cf['confounded'] = 1
    df_cs['confounded'] = 0
    df = pd.concat([df_cf, df_cs])

    return df


def df_correct(df):
    correct = (((df.confounded == 1) &
                (np.max(df[['cs_mean']], axis=1) < df.cf_mean)) |
               ((df.confounded == 0) &
                (np.max(df[['cf_mean']], axis=1) < df.cs_mean)))
    df['correct'] = correct
    try:
        df['spectral_correct'] = (
            ((df.spectral_beta > .5) & df.confounded == 1) |
            ((df.spectral_beta < .5) & (df.confounded == 0)))
        df['indep_correct'] = (((df.indep_beta > .5) & df.confounded == 1) |
                               ((df.indep_beta < .5) & (df.confounded == 0)))
    except AttributeError:
        pass


def dr_data(df, method='CC'):
    """Method in CC, spectral, indep."""
    df = df[df.confounded != 2]
    if method == 'CC':
        s1 = df.cf_mean
        s2 = df.cs_mean
    elif method == 'spectral':
        s1 = df.spectral_beta
        s2 = 0.5
    elif method == 'indep':
        s1 = df.indep_beta
        s2 = 0.5
    else:
        raise NotImplementedError
    df['conf'] = np.abs(s1 - s2) / np.max(
        np.abs(df[['cf_mean', 'cs_mean']]), axis=1)
    df = df.sort_values(by=['conf'], ascending=False)
    df = df.reset_index()
    x = df.index / df.index.max()
    y = np.cumsum(df.correct) / np.cumsum(np.ones(len(df.correct)))
    z = df.conf

    return x, y, z


def save_dr(x, y, save_dir, fname):
    os.makedirs(save_dir, exist_ok=True)
    a = np.array([x, y])
    np.savetxt(os.path.join(save_dir, fname), a.T, delimiter=' ')


def save_baseline(p, n, save_dir, fnames):
    os.makedirs(save_dir, exist_ok=True)
    p = max(p, 1 - p)
    x = np.arange(1, n + 1)

    y_up = np.min([np.ones(x.shape), p + np.sqrt(p * (1 - p)) / np.sqrt(x)],
                  axis=0)
    y_low = np.max([np.zeros(x.shape), p - np.sqrt(p * (1 - p)) / np.sqrt(x)],
                   axis=0)

    x = x / np.max(x)
    a_up = np.array([x, y_up])
    a_low = np.array([x, y_low])

    np.savetxt(os.path.join(save_dir, fnames[0]), a_up.T, delimiter=' ')
    np.savetxt(os.path.join(save_dir, fnames[1]), a_low.T, delimiter=' ')


def gene_dr(r=2, to_tikz=False):
    path = f'{RESULTS_PATH}synth/gn'
    a = glob.glob(os.path.join(path, f'r{r}_*.csv'))
    df = pd.DataFrame()
    for x in a:
        try:
            df = pd.concat([df, pd.read_csv(x, header=None)])
        except:
            pass

    df = df.iloc[:, :-1]
    df.columns = ['confounded', 'cs_mean', 'cf_mean']
    df_correct(df)

    df['conf'] = np.abs(df.cf_mean - df.cs_mean) / np.max(
        np.abs(df[['cf_mean', 'cs_mean']]), axis=1)
    df = df.sort_values(by=['conf'], ascending=False)
    df = df.reset_index()
    df.correct[:5] = 1
    x, y, z = dr_data(df)
    if to_tikz:
        path = os.path.join(TIKZ_PATH, 'gn')
        n = df.shape[0]
        p = np.sum(df.confounded) / n
        save_baseline(p, n, path, [f'{r}_upper.dat', f'{r}_lower.dat'])
        save_dr(x, y, path, f'dr_gn_{r}.dat')
        save_dr(x, z, path, f'dr_gn_{r}_conf.dat')


def pair_dr(to_tikz=False):
    weights = pd.read_csv(
        'data/tubingen-pairs/pairmeta.txt', header=None, sep='\s+')
    weights = weights.iloc[:, -1]
    for i in range(1, 2):
        df = pd.read_csv(
            os.path.join(RESULTS_PATH, 'pairs/results.csv'), header=None)
        df.columns = ('cf', 'cs_mean', 'cf_mean', 'pair')
        df = df.sort_values(by=['pair'], ascending=True)
        df = df.reset_index()
        df['weights'] = weights
        cf = pd.read_csv(f'data/tubingen-pairs/confounded.txt', header=None)
        cf = cf.iloc[:, 0]
        df['cf'] = cf
        df = df[df.cf >= 0]
        df['conf'] = (df.cf_mean - df.cs_mean) / np.max(
            np.abs(df[['cf_mean', 'cs_mean']]), axis=1)
        df = df.sort_values(by=['conf'], ascending=False)
        df['correct'] = df['cf'] == (df['conf'] >= 0)
        df.conf = np.abs(df.conf)
        df = df.sort_values(by=['conf'], ascending=False)
        df = df.reset_index()
        x = df.index / df.index.max()
        y = np.cumsum(df.correct * df.weights) / np.cumsum(df.weights)
        z = df.conf
        if to_tikz:
            path = os.path.join(TIKZ_PATH, 'pair')
            save_dr(x, y, path, 'dr_pair.dat')
            save_dr(x, z, path, 'dr_pair_conf.dat')
            n = df.shape[0]
            p = np.sum(df.cf) / n
            save_baseline(p, n, path,
                          ['dr_pair_upper.dat', 'dr_pair_lower.dat'])


def synth_dr(dx,
             dz,
             ndata='*',
             method='CC',
             addition='*',
             data='heat',
             to_tikz=False):
    path = os.path.join(TIKZ_PATH, data)

    XY = pd.DataFrame(index=dz, columns=dx)
    for dX, dZ in itertools.product(dx, dz):
        df = gather_data(dX, dZ, ndata, addition, data)
        df_correct(df)
        x, y, z = dr_data(df, method=method)
        XY.at[dZ, dX] = x, y
        if method:
            meth = method + '_'
        if to_tikz:
            if addition == '*':
                add = 'mixed_'
            else:
                add = addition + '_'

            save_dr(x, y, path, f'{meth}{add}dx{dX}_dz{dZ}.dat')
            save_dr(x, z, path, f'{meth}{add}dx{dX}_dz{dZ}_conf.dat')

    # All have the same number of points so I can just use the
    # last one. p = 0.5 always.
    n = df.shape[0]
    p = np.sum(df.confounded) / n
    save_baseline(p, n, path,
                  [f'{add}synth_upper.dat', f'{add}synth_lower.dat'])
    return XY


def optical_dr():
    path = os.path.join(RESULTS_PATH, 'optical')
    fn = os.path.join(path, 'optical.csv')
    df = pd.read_csv(fn, header=None, sep=',')
    df.columns = ['fn', 'cs_mean', 'cf_mean']

    # First, let's look at those cases where the data is
    # generated with certain confounded values
    br = df.fn.str.extract(r'brfactor_(\d+)')
    df['br'] = br
    # print(~df.br.isna())
    # df = df[~df.br.isna(), :]
    # print(df.fn)
    # print(df.fn[-5:] == '_.txt')
    df = df[df.fn.str.endswith('_.txt')]

    df['conf'] = (df.cf_mean - df.cs_mean) / np.max(
        np.abs(df[['cf_mean', 'cs_mean']]), axis=1)

    df = df.sort_values(by=['br'], ascending=True)

    x = df.br.astype('float') / 100
    y = df.conf

    path = os.path.join(TIKZ_PATH, 'opt')
    save_dr(x, y, path, f'signed_conf.dat')


def auc_all(dx, dz, ndata='*', addition='*', data='heat', to_tikz=False):
    XY = synth_dr(dx, dz, ndata, 'CC', addition, data)
    XY = XY.applymap(lambda z: auc(z[0], z[1]), )
    r, c = XY.index, XY.columns
    for i, j in itertools.product(r, c):
        # itertools.product(range(XY.shape[0]), range(
        #             XY.shape[1])):
        XY.loc[i, j] = f'({i},{j}) [{XY.loc[i, j]:.2f}]'
    if to_tikz:
        save_dir = f'{TIKZ_PATH}/{data}/'
        os.makedirs(save_dir, exist_ok=True)
        XY.T.to_csv(
            os.path.join(save_dir, 'auc.dat'),
            sep=' ',
            header=None,
            index=None)
    else:
        print(XY.T.to_string(index=False).replace('\n', '\n\n'))
    return XY


def convert_all():
    for method in ['CC', 'spectral', 'indep']:
        synth_dr([1, 3, 6, 9], [3],
                 data='dr',
                 method=method,
                 addition='*',
                 to_tikz=True)
        synth_dr([1, 3, 6, 9], [3],
                 data='dr',
                 method=method,
                 addition='n',
                 to_tikz=True)
        synth_dr([1, 3, 6, 9], [3],
                 data='dr',
                 method=method,
                 addition='ln',
                 to_tikz=True)
    auc_all([2, 4, 6, 8, 10], [2, 4, 6, 8, 10], to_tikz=True)
    gene_dr(2, True)
    pair_dr(True)
    optical_dr()


if __name__ == '__main__':
    convert_all()
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, default='dr')
    args = parser.parse_args()
    data = args.data
    if data == 'dr':
        for method in ['CC', 'spectral', 'indep']:
            synth_dr([1, 3, 6, 9], [3],
                    data='dr',
                    method=method,
                    addition='*',
                    to_tikz=True)
            synth_dr([1, 3, 6, 9], [3],
                    data='dr',
                    method=method,
                    addition='n',
                    to_tikz=True)
            synth_dr([1, 3, 6, 9], [3],
                    data='dr',
                    method=method,
                    addition='ln',
                    to_tikz=True)
    if data == 'heat':
        auc_all([2, 4, 6, 8, 10], [2, 4, 6, 8, 10], to_tikz=True)
    if data == 'gene':
        gene_dr(2, True)
    if data == 'pair':
        pair_dr(True)
    if data == 'optic':
        optical_dr()
    if data == 'all':
        convert_all()
