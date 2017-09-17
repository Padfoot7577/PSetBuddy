import pickle
import re
import os

from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.patches as pat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import do_clustering as cluster


def fit_k(Model, data, k_min, k_max, snap_path, verbose, **model_opts):
    for k in range(k_min, k_max + 1):
        while True:
            print('Fitting k = %d' % k, end=': ', flush=True)
            model = Model(k, **model_opts)
            if model.fit(data, verbose=verbose): # verbose=False
                break
            print('bad init; trying again...')
        model_type = Model.__name__.lower()
        msnap_path = os.path.join(snap_path, '%s_k%s.pkl' % (model_type, k))
        with open(msnap_path, 'wb') as f_snap:
            pickle.dump(model, f_snap)

def get_k(p):
    return int(re.search('_k(\d+)\.pkl', p).group(1))

def load_categories(catfile):
    with open(catfile) as f_cat:
        field_cats = {}
        cur_field = None
        for l in f_cat:
            l = l.rstrip()
            if not l:
                continue
            elif not l.startswith(' '):
                field, field_desc = l.split(' - ')
                cur_field = field
                field_cats[cur_field] = []
            else:
                field_cats[cur_field].append(l.split(': ')[1])
    return field_cats

def print_clusters(model, fields, categories):
    assert isinstance(model, cluster.CMM)
    assert len(model.alpha) == len(fields)
    assert len(fields) <= len(categories)

    max_cats = np.zeros((model.k, len(fields)))
    for i, a in enumerate(model.alpha):
        max_cats[:, i] = a.argmax(1)

    for k in range(model.k):
        cluster_mcs = max_cats[k].astype(int)
        cnames = [categories[f][cluster_mcs[i]] for i, f in enumerate(fields)]
        fc_strs = '\n  '.join('%s: %s' % fc for fc in zip(fields, cnames))
        print('Cluster %s:\n  %s\n' % (k + 1, fc_strs))

