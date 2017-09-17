import glob
import os
import pickle

import do_clustering as cluster
import utils as utils

import numpy as np
import pandas as pd

import time
import random


# Cluster
def clusterize():

    PROJ_DIR = os.path.abspath(os.path.dirname(__file__))
    MODELS_DIR = os.path.join(PROJ_DIR, 'models')
    
    if not os.path.exists(MODELS_DIR):
        os.mkdir(MODELS_DIR)

#start = time.time()
  
#field_cats = utils.load_categories(os.path.join(PROJ_DIR, 'categories.txt'))
    data = pd.read_csv(os.path.join(PROJ_DIR, 'student_data_for_import.csv'))
# print(data)
    ds = data.apply(pd.Series.nunique)
  
    CMM_K_MIN_MAX = (18, 18) # (2, 20)
    utils.fit_k(cluster.CMM, data, *CMM_K_MIN_MAX, MODELS_DIR, verbose=True, ds=ds)
  
#end = time.time()
#print("Runtime:", end - start)


# Choose from Cluster
# Beware of off-by-one errors! database's 0th row contains category labels!
def find_buddy(database_index):
    PROJ_DIR = os.path.abspath(os.path.dirname(__file__))
    MODELS_DIR = os.path.join(PROJ_DIR, 'models')
    snaps = glob.glob(os.path.join(MODELS_DIR, 'cmm_*.pkl'))
    snaps.sort(key=utils.get_k)
    snap = snaps[-3]
    with open(snap, 'rb') as f_snap:
        model = pickle.load(f_snap)
        all_students_cluster_ids = np.argmax(model.params["p_z"], axis=1)
    #    print(np.argmax(model.params["p_z"], axis=1))
    #    print(np.unique(np.argmax(model.params["p_z"], axis=1)))
    cluster_assignment = all_students_cluster_ids[database_index]
    buddies = []
    for i, cluster_id in enumerate(all_students_cluster_ids):
        if cluster_id == cluster_assignment:
            buddies.append(i)
    if len(buddies) == 1:
        guess = random.randint(0, len(cluster_assignment)-1)
        while guess == database_index:
            guess = random.randint(0, len(cluster_assignment)-1)
    else:
        guess = random.choice(buddies)
    return guess # return the index of the buddy in the database


# Analyze Cluster (TODO)
# K_SHOW = 18  # best K and then some other k
# with open(os.path.join(MODELS_DIR, 'cmm_k%d.pkl' % K_SHOW), 'rb') as f_model:
#     model = pickle.load(f_model)
# utils.print_clusters(model, data.columns, field_cats)

# start = time.time()
#   
# field_cats = utils.load_categories(os.path.join(PROJ_DIR, 'categories.txt'))
# data = pd.read_csv(os.path.join(PROJ_DIR, 'student_data_for_import.csv'))
# ds = data.apply(pd.Series.nunique)
#   
# CMM_K_MIN_MAX = (9, 9)
# utils.fit_k(cluster.CMM, data, *CMM_K_MIN_MAX, MODELS_DIR, verbose=False, ds=ds)
#   
# end = time.time()
# print("Runtime:", end - start)
