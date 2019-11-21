import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
from utils import *

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

pjoin = os.path.join


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--run_root', default='/Users/yefeichen/Database/location_recommender_system/')
    arg('--app_date',default='_191113')
    arg('--prev_name',type=str,default='sampled_ww_')
    args = parser.parse_args()

    datapath = args.run_root
    cfile = ['dnb_pa.csv', 'dnb_sf.csv', 'dnb_sj.csv', 'dnb_Los_Angeles.csv', 'dnb_New_York.csv']
    app_date = args.app_date
    apps = app_date + '.csv'

    citynameabbr = ['PA','SF','SJ','LA','NY']
    clfile = [ c+apps for c in citynameabbr ]
    comp_feat_file = 'company_feat' + apps
    ssfile = [ args.prev_name + c + '_similarity'+apps for c in citynameabbr]

    print('Loading company normalized feature...')
    comp_feat = pd.read_csv(pjoin(datapath, comp_feat_file), index_col=0)
    comp_feat_col = [c for c in comp_feat.columns if c not in ['duns_number', 'atlas_location_uuid']]

    for ind_city in range(len(citynameabbr)):
        # atlas_location_uuid duns_number similarity
        print('Processing city: %s' % citynameabbr[ind_city])
        pred_dat = pd.read_csv(pjoin(datapath, ssfile[ind_city]), index_col=0)
        gr_dat = pd.read_csv(pjoin(datapath, clfile[ind_city]))
        pred_gr_dat = pred_dat.merge(gr_dat[['atlas_location_uuid', 'duns_number']], on=['atlas_location_uuid'],
                                     how='left', suffixes=['_prd', '_grd'])

        print('Pairs to be calced:%d' % len(pred_gr_dat))
        print('Preparing feature...')
        prd_comp_feat = \
            pred_gr_dat[['duns_number_prd']].rename(columns={'duns_number_prd': 'duns_number'}) \
                .merge(comp_feat, on='duns_number', how='left')[comp_feat_col].to_numpy()
        grd_comp_feat = \
            pred_gr_dat[['duns_number_grd']].rename(columns={'duns_number_grd': 'duns_number'}) \
                .merge(comp_feat, on='duns_number', how='left')[comp_feat_col].to_numpy()

        print('Doing cosine distance...(1-xy/nx/ny)')
        prd_comp_feat = normalize(prd_comp_feat, axis=1)
        grd_comp_feat = normalize(grd_comp_feat, axis=1)
        dist = 1 - (prd_comp_feat * grd_comp_feat).sum(axis=1).reshape(-1, 1)

        distpd = pd.DataFrame(dist, columns=['dist'])

        pred_gr_dat2 = pd.concat([pred_gr_dat[['atlas_location_uuid', 'duns_number_prd', 'duns_number_grd']], distpd],
                                 axis=1)
        #in case of the identical company inside
        pred_gr_dat2.loc[pred_gr_dat2['dist'] < 1e-12, 'dist'] = 1
        result = pred_gr_dat2.groupby(['atlas_location_uuid', 'duns_number_prd'])['dist'].min().reset_index()

        num_location = len(result.groupby('atlas_location_uuid').first().reset_index())

        print('===============Statistic Board====================')
        print('# city: %s'%citynameabbr[ind_city])
        print('# number of location:%d'%num_location)
        print('# number of queries %d'%len(pred_gr_dat))
        print('# min-dist of each queries:')
        print(result['dist'].describe())
        print('==================================================')
