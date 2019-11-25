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

pjoin = os.path.join

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--run_root', default='/Users/yefeichen/Database/location_recommender_system/')
    arg('--app_date',default='_191114')
    arg('--ratio',default=0.8)

    args = parser.parse_args()
    max_K = 20

    apps = args.app_date + '.csv'

    citynameabbr = ['PA', 'SF', 'SJ', 'LA', 'NY']
    datapath = args.run_root

    clfile = [c + apps for c in citynameabbr]

    saveTrname = [ c+'_train'+apps for c in citynameabbr]
    saveTtname = [ c+'_test'+apps for c in citynameabbr]

    trdats = []
    ttdats = []

    for ind_city,filename in enumerate(clfile):
        print('Processing city: %s'%filename)
        cldat = pd.read_csv(pjoin(datapath,filename))
        #unique location list
        locdat = cldat.groupby('atlas_location_uuid').first().reset_index()
        trlocdat = locdat.sample(frac=args.ratio).reset_index(drop=True)
        ttlocdat = locdat.merge(trlocdat, on=['atlas_location_uuid'], how='left', suffixes=['', '_right'])
        ttlocdat = ttlocdat[ttlocdat['geo_distance_right'].isnull()].reset_index()
        trlocdat = trlocdat[['atlas_location_uuid']]
        ttlocdat = ttlocdat[['atlas_location_uuid']]
        print('location for train: %d, for test: %d '%(len(trlocdat),len(ttlocdat)))
        #attach label
        trlocdat['fold'] = 0
        ttlocdat['fold'] = 2

        #attach label into location-company pairs
        trttlocdat = pd.concat([trlocdat, ttlocdat], axis=0).reset_index(drop=True)
        cldat = cldat.merge(trttlocdat, on='atlas_location_uuid', how='left', suffixes=['', '_right'])[['atlas_location_uuid','duns_number','fold']]
        cldat['city'] = ind_city
        print('location-company pairs for train: %d, for test %d'%(len(cldat[cldat['fold']==0]),len(cldat[cldat['fold']==2])))

        #saveing trdats
        trdats.append(cldat)

        #operate on ttdats
        cldat = cldat[cldat['fold'] == 2]
        fn = lambda obj: obj.loc[np.random.choice(obj.index, 1, True),:]
        tbA = cldat.groupby('atlas_location_uuid').apply(fn).reset_index(drop=True)[['duns_number', 'atlas_location_uuid']]
        print('1.len of tbA %d:'%len(tbA))
        fn = lambda obj: obj.loc[np.random.choice(obj.index, max_K, True),:]
        tbB = cldat.groupby('atlas_location_uuid').apply(fn).reset_index(drop=True)[['duns_number', 'atlas_location_uuid']]
        print('1.len of tbB %d'%len(tbB))

        ###======================Pos=============================###
        tbA['mk'] = 'A'
        tbB = tbB.merge(tbA,on=['duns_number','atlas_location_uuid'],how='left',suffixes=['','_right'])
        tbB = tbB[tbB['mk'].isnull()]
        print('2.len of tbB not included in tbA %d'%len(tbB))
        #we need to full fill the data
        tbB = tbB.groupby('atlas_location_uuid').apply(fn).reset_index(drop=True)[['duns_number', 'atlas_location_uuid']]
        tbB['mk'] = 'B'
        print('3.len of tbB full filled again %d'%len(tbB))
        #in case tbB cut some locations from tbA, lets shrink tbA
        tblocB = tbB.groupby('atlas_location_uuid').first().reset_index()
        print('4.len of locations in tbB %d'%len(tblocB))
        tbA = tbA.merge(tblocB,on='atlas_location_uuid',how='left',suffixes=['','_right'])
        tbA = tbA[tbA['mk_right'].notnull()][['duns_number', 'atlas_location_uuid','mk']].reset_index(drop=True)
        print('4.len of tbA with common locations of tbB %d'%len(tbA))

        ###======================Neg=============================###
        tbAA = pd.concat([tbA,tbA.sample(frac=1).reset_index()\
                   .rename(columns={'duns_number':'duns_number_n','atlas_location_uuid':'atlas_location_uuid_n','mk':'mk_n'})]
                  ,axis=1)
        print('5.len of negpair %d'%len(tbAA))
        tbAA = tbAA.merge(cldat,\
                   left_on=['duns_number_n','atlas_location_uuid'],right_on=['duns_number','atlas_location_uuid'],\
                   how='left', suffixes = ['','_right'])

        tbC = tbAA[tbAA['duns_number_right'].isnull()][['duns_number_n','atlas_location_uuid']]\
                .rename(columns={'duns_number_n':'duns_number'})
        print('6.len of neg data %d'%len(tbC))

        #in case tbC cut some locations from tbA and tbB
        tbC['mk'] = 'C'
        tblocC = tbC.groupby('atlas_location_uuid').first().reset_index()
        print('6.locations in neg data %d'%len(tblocC))
        tbA = tbA.merge(tblocC,on='atlas_location_uuid',how='left',suffixes=['','_right'])
        tbA = tbA[tbA['mk_right'].notnull()][['duns_number', 'atlas_location_uuid','mk']].reset_index(drop=True)
        print('final tbA len %d'%len(tbA))

        tbB = tbB.merge(tblocC,on='atlas_location_uuid',how='left',suffixes=['','_right'])
        tbB = tbB[tbB['mk_right'].notnull()][['duns_number', 'atlas_location_uuid','mk']].reset_index(drop=True)
        print('final tbB len %d'%len(tbB))

        tbA = tbA.sort_values(by='atlas_location_uuid')
        tbB = tbB.sort_values(by='atlas_location_uuid')
        tbC = tbC.sort_values(by='atlas_location_uuid')

        assert(len(tbA)==len(tbC) and len(tbB)==len(tbA)*max_K)

        result = pd.concat([tbA, tbB, tbC], axis=0).reset_index(drop=True)
        result['city'] = ind_city
        print(len(result))

        ttdats.append(result)


    trdats = pd.concat(trdats,axis=0).reset_index(drop=True)
    ttdats = pd.concat(ttdats,axis=0).reset_index(drop=True)

    trdats.to_csv('region_train.csv')
    ttdats.to_csv('region_test.csv')











