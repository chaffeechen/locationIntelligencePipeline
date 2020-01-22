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

from header import *
from dnb.data_loader import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--run_root', default='/Users/yefeichen/Database/location_recommender_system/')
    arg('--ls_card',default='location_scorecard_191113.csv')
    arg('--app_date',default='_191114')
    arg('--dbname',default='tmp_table')
    args = parser.parse_args()

    datapath = args.run_root
    datapath_mid = pjoin(datapath,args.dbname)

    app_date = args.app_date
    apps = app_date + '.csv'
    appsadd = app_date + '_add.csv'

    dataloader = data_process(root_path=datapath)
    dnb_city_lst = dataloader.load_dnb_city_lst(db=args.dbname,table='dnb_city_list'+apps)
    """
    "citylongname":citylongname,
    "cityabbr":cityabbr,
    "origin_comp_file":origin_comp_file,
    """

    cfile = dnb_city_lst['origin_comp_file']
    clfile = dnb_city_lst['cityabbr']

    lfile = args.ls_card  # It is fixed as input
    clfile = [c + apps for c in clfile]

    bid = 'atlas_location_uuid'
    cid = 'duns_number'

    print('Args:',datapath,apps,lfile)

    not_feat_col = feature_column['not_feat_col']
    cont_col_nameC = feature_column['cont_col_nameC']
    spec_col_nameC = feature_column['spec_col_nameC']
    cont_col_nameL = feature_column['cont_col_nameL']
    key_col_comp = feature_column['key_col_comp']
    key_col_loc = feature_column['key_col_loc']

    dummy_col_nameL = feature_column['dummy_col_nameC']
    dummy_col_nameC = feature_column['dummy_col_nameL']


    ##Multi training data generator(multi city)
    # 如果不合并所有数据在进行dummy 会出现一些category在某些城市不出现的情况，从而导致问题
    # 8-2分训练测试集

    train_test_val_pairs = []
    dat_comp_pds = []
    dat_loc_pds = []

    pdlls = []  # all location feat pd list
    pdccs = []

    pdl = pd.read_csv(pjoin(datapath, lfile),index_col=0)
    for ind_city in range(len(cfile)):
        pdc = pd.read_csv(pjoin(datapath, cfile[ind_city]))
        pdccs.append(pdc)

    # for loop end
    # building feature
    # company feature
    # pdlls = pd.concat(pdlls, axis=0).reset_index(drop=True)
    pdlls = pdl.reset_index()
    pdccs = pd.concat(pdccs, axis=0).reset_index(drop=True)

    print('start processing company and location feature...')

    # one hot explanation
    comp_one_hot_col_name = dummy_col_nameC #['major_industry_category', 'location_type', 'primary_sic_2_digit']
    loc_one_hot_col_name = dummy_col_nameL #['building_class']

    print('one hot description loading...')
    comp_coldict = load_obj(pjoin(datapath_mid, 'comp_feat_dummy_param' + app_date))
    loc_coldict = load_obj(pjoin(datapath_mid, 'loc_feat_dummy_param' + app_date))

    print('dummy...')
    XD_loc,dummy_onehot_nameL = apply_dummy(coldict=loc_coldict, data=pdlls)
    XD_comp,dummy_onehot_nameC = apply_dummy(coldict=comp_coldict, data=pdccs)


    print('normalization descriptor loading...')
    comp_norm_param = load_obj(pjoin(datapath_mid, 'comp_feat_norm_param' + app_date))
    loc_norm_param = load_obj(pjoin(datapath_mid, 'loc_feat_norm_param' + app_date))

    print('normalization...')
    pdccs['city'] = pdccs['physical_city']
    cont_comp = comp_dat_process(pdccs, one_hot_col_name=dummy_col_nameC,cont_col_name=cont_col_nameC,\
                                 spec_col_name=spec_col_nameC, do_dummy=False)
    cont_loc = location_dat_process(pdlls, one_hot_col_name=dummy_col_nameL,cont_col_name=cont_col_nameL, \
                                    do_dummy=False)

    XC_comp = apply_para_normalize_dat(cont_comp[cont_col_nameC], comp_norm_param['C_comp'],
                                              comp_norm_param['S_comp'])
    XC_loc = apply_para_normalize_dat(cont_loc['data'][cont_col_nameL], loc_norm_param['C_loc'],
                                             loc_norm_param['S_loc'])

    y_comp_name = key_col_comp
    y_comp_name.append('city')
    y_loc_name = key_col_loc
    y_loc_name.append('city')

    assert('city' in pdlls.columns)
    Y_loc = pdlls[y_loc_name].fillna('none').to_numpy()
    Y_comp = pdccs[y_comp_name].to_numpy()

    print('concating')
    X_comp = np.concatenate([Y_comp, XC_comp, XD_comp], axis=1)
    X_loc = np.concatenate([Y_loc, XC_loc, XD_loc], axis=1)

    c_comp_name = cont_col_nameC
    d_comp_name = dummy_onehot_nameC
    c_loc_name = cont_col_nameL
    d_loc_name = dummy_onehot_nameL

    print('numpy 2 pandas')
    dat_comp_pd = pd.DataFrame(data=X_comp, columns=y_comp_name + c_comp_name + d_comp_name)
    dat_loc_pd = pd.DataFrame(data=X_loc, columns=y_loc_name + c_loc_name + d_loc_name)

    print(dat_comp_pd.to_numpy()[:,2:].mean())
    print(dat_loc_pd.to_numpy()[:, 2:].mean())
    print(dat_comp_pd.shape)

    print('Done')

    # print('Final merge...')
    dat_comp_pd.to_csv(pjoin(datapath_mid, 'company_feat' + appsadd))
    dat_loc_pd.to_csv(pjoin(datapath_mid, 'location_feat' + appsadd))
    print('All Done')

    print(dat_comp_pd.shape,dat_loc_pd.shape)