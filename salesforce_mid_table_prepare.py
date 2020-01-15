import pandas as pd
import numpy as np

import os,sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
from header import *

sfx = ['','_right']

pj = os.path.join

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--run_root', default='/home/ubuntu/location_recommender_system/')
    arg('--ls_card',default='location_scorecard_200106.csv')
    arg('--app_date',default='_200114')
    args = parser.parse_args()

    cid = salesforce_space['cid']
    bid = salesforce_space['bid']

    datapath = args.run_root
    sfpath = pj(datapath,'salesforce')

    """
    Output db name
    """
    acc_city_db_name = salesforce_space['acc_city_db'] + args.app_date + '.csv'
    city_atlas_db_name = salesforce_space['city_atlas_db'] + args.app_date + '.csv'

    print('==> loading data')
    acc_db = pd.read_csv(pj(sfpath, salesforce_space['acc_db']), index_col=0)
    visit_db = pd.read_csv(pj(sfpath, salesforce_space['visit_db']), index_col=0)
    loc_db = pd.read_csv(pj(datapath, args.ls_card), index_col=0)

    print('==> De-duplication')
    de_visit_db = visit_db.groupby([cid,bid]).first().reset_index()
    print('%d vs. %d = %1.2f'%(len(de_visit_db),len(visit_db),len(de_visit_db)/len(visit_db)))

    acc_city_loc_db = de_visit_db.merge(loc_db[[bid, 'city']])[[cid, bid, 'city']]

    # Acc_city_table
    print('==> acc x city table')
    acc_city_db = acc_city_loc_db.groupby([cid, 'city']).first().reset_index()[[cid, 'city']]
    acc_city_db.to_csv(pj(sfpath, acc_city_db_name ))

    # City_atlas_table
    print('==> city x atlas table')
    city_atlas_db = acc_city_loc_db.groupby([bid, 'city']).first().reset_index()[[bid, 'city']]
    city_atlas_db.to_csv(pj(sfpath, city_atlas_db_name))

    print('Done')
