#describe each location with companies in side

import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


import pandas as pd
import numpy as np
import pickle
import argparse
from math import *
from sklearn.preprocessing import normalize
from utils import *

from header import *

pjoin = os.path.join

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--run_root', default='/Users/yefeichen/Database/location_recommender_system/')
    arg('--ls_card',default='location_scorecard_191113.csv')
    arg('--app_date',default='_191114')
    arg('--ratio',type=float,default=0.8)
    arg('--outfile',default='train_val_test_location_company_82split_191113.csv')
    args = parser.parse_args()

    datapath = args.run_root
    cfile = origin_comp_file
    app_date = args.app_date
    apps = app_date + '.csv'
    lfile = args.ls_card  # It is fixed as input
    clfile = [c + apps for c in cityabbr]

    print('Args:',datapath,apps,lfile,args.ratio)

    colname = {
        'company': 'duns_number',
        'location': 'atlas_location_uuid'
    }

    key_column = [colname['company'], colname['location']]

    train_test_val_pairs = []

    # pdlls = []  # all location feat pd list
    # pdccs = []
    for ind_city in range(len(clfile)):
        # pdc = pd.read_csv(pjoin(datapath, cfile[ind_city]))
        # pdl = pd.read_csv(pjoin(datapath, lfile))
        pdcl = pd.read_csv(pjoin(datapath, clfile[ind_city]))

        print('generating train_val_test csv')
        # train_test_val_pairs :[ duns_number, atlas_location_uuid, label, city, fold ]

        #pair_dat = [ duns_number, atlas_location_uuid, label]
        pair_dat = getPosNegdatv2_fast_general(pdcl,colname=colname)

        tr, tt = splitdat(pair_dat, key_column=key_column, right_colunm='label_tr',
                          rate_tr=args.ratio)
        # training pair ==> pair format with positive only

        pot_pos_dat = pdcl[key_column]
        pot_pos_dat = pd.merge(pot_pos_dat,tt,on=key_column,how='left',suffixes=['','_right'])
        train_pos_pair = pot_pos_dat[pot_pos_dat['label'].isnull()]
        train_pos_pair['label'] = 1
        ## ATT need acceleration!!!!!
        # testing pair ==> pair format with positive and negative both
        testing_pair = tt.reset_index()[ key_column.append('label') ]

        train_pos_pair['fold'] = 0
        testing_pair['fold'] = 2

        train_test_val_pair = pd.concat([train_pos_pair, testing_pair])
        train_test_val_pair['city'] = ind_city
        train_test_val_pairs.append(train_test_val_pair)
        print(len(train_test_val_pair))
        print('train_val_test_location_company Done')

    train_test_val_pair = pd.concat(train_test_val_pairs)
    train_test_val_pair.to_csv(pjoin(datapath, args.outfile ))

    print('All Done')