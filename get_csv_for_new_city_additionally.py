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
    arg('--ls_card',default='location_scorecard_191113.csv')
    arg('--app_date',default='_191114')
    arg('--ratio',type=float,default=0.8)
    args = parser.parse_args()

    datapath = args.run_root
    cfile = ['dnb_pa.csv']
    app_date = args.app_date
    apps = app_date + '.csv'
    appsadd = app_date+'_add'+'.csv'
    lfile = args.ls_card  # It is fixed as input
    clfile = ['PA']
    clfile = [c + apps for c in clfile]

    print('Args:',datapath,apps,lfile,args.ratio)

    not_feat_col = ['duns_number',
                    'atlas_location_uuid',
                    'longitude_loc',
                    'latitude_loc',
                    'label']
    cont_col_nameC = ['emp_here', 'emp_total', 'sales_volume_us', 'square_footage', 'emp_here_range'] #make sure that spec_col_nameC is in the end of line!!!
    spec_col_nameC = 'emp_here_range'
    cont_col_nameL = ['score_predicted_eo', 'score_employer', 'num_emp_weworkcore', 'num_poi_weworkcore',
                      'pct_wwcore_employee', 'pct_wwcore_business', 'num_retail_stores', 'num_doctor_offices',
                      'num_eating_places', 'num_drinking_places', 'num_hotels', 'num_fitness_gyms',
                      'population_density', 'pct_female_population', 'median_age', 'income_per_capita',
                      'pct_masters_degree', 'walk_score', 'bike_score']
    key_col_comp = ['duns_number']
    key_col_loc = ['atlas_location_uuid']

    dummy_col_nameL = ['building_class']
    dummy_col_nameC = ['major_industry_category', 'location_type', 'primary_sic_2_digit']


    ##Multi training data generator(multi city)
    # 如果不合并所有数据在进行dummy 会出现一些category在某些城市不出现的情况，从而导致问题
    # 8-2分训练测试集

    train_test_val_pairs = []
    dat_comp_pds = []
    dat_loc_pds = []

    pdlls = []  # all location feat pd list
    pdccs = []
    for ind_city in range(len(cfile)):
        pdc = pd.read_csv(pjoin(datapath, cfile[ind_city]))
        pdl = pd.read_csv(pjoin(datapath, lfile))
        pdcl = pd.read_csv(pjoin(datapath, clfile[ind_city]))

        print('generating train_val_test csv')
        # train_test_val_pairs :[ duns_number, atlas_location_uuid, label, city, fold ]
        pair_dat = getPosNegdatv2_fast(pdcl)
        tr, tt = splitdat(pair_dat, key_column=['duns_number', 'atlas_location_uuid'], right_colunm='label_tr',
                          rate_tr=args.ratio)
        # training pair ==> pair format with positive only

        pot_pos_dat = pdcl[['duns_number', 'atlas_location_uuid']]
        pot_pos_dat = pd.merge(pot_pos_dat,tt,on=['duns_number', 'atlas_location_uuid'],how='left',suffixes=['','_right'])
        train_pos_pair = pot_pos_dat[pot_pos_dat['label'].isnull()]
        train_pos_pair['label'] = 1
        ## ATT need acceleration!!!!!
        # train_pos_pair = \
        # tr[tr['label'] == 1].groupby(['duns_number', 'atlas_location_uuid', 'label']).first().reset_index()[
        #     ['duns_number', 'atlas_location_uuid', 'label']]

        # testing pair ==> pair format with positive and negative both
        testing_pair = tt.reset_index()[['duns_number', 'atlas_location_uuid', 'label']]

        train_pos_pair['fold'] = 0
        testing_pair['fold'] = 2

        train_test_val_pair = pd.concat([train_pos_pair, testing_pair])
        train_test_val_pair['city'] = ind_city
        train_test_val_pairs.append(train_test_val_pair)
        print(len(train_test_val_pair))
        print('train_val_test_location_company Done')

        # building features
        col_list = list(pdl.columns)
        pdll = pdl.merge(pdcl, how='inner', on=['atlas_location_uuid'], suffixes=['', '_right'])
        pdll = pdll[pdll['duns_number'].isnull() == False]
        pdll = pdll.groupby(['atlas_location_uuid']).first().reset_index()
        pdll = pdll[col_list]
        pdlls.append(pdll)

        # company feature
        pdccs.append(pdc)

    # for loop end
    # building feature
    # company feature
    pdlls = pd.concat(pdlls, axis=0).reset_index(drop=True)
    pdccs = pd.concat(pdccs, axis=0).reset_index(drop=True)

    print('start processing company and location feature...')

    # one hot explanation
    comp_one_hot_col_name = dummy_col_nameC #['major_industry_category', 'location_type', 'primary_sic_2_digit']
    loc_one_hot_col_name = dummy_col_nameL #['building_class']

    print('one hot description loading...')
    comp_coldict = load_obj(pjoin(datapath, 'comp_feat_dummy_param' + app_date))
    loc_coldict = load_obj(pjoin(datapath, 'loc_feat_dummy_param' + app_date))

    print('dummy...')
    XD_comp = apply_dummy(coldict=comp_coldict, data=pdccs)
    XD_loc = apply_dummy(coldict=loc_coldict, data=pdlls)

    print('normalization descriptor loading...')
    comp_norm_param = load_obj(pjoin(datapath, 'comp_feat_norm_param' + app_date))
    loc_norm_param = load_obj(pjoin(datapath, 'loc_feat_norm_param' + app_date))

    print('normalization...')

    cont_comp = comp_dat_process(pdccs, one_hot_col_name=dummy_col_nameC,cont_col_name=cont_col_nameC,\
                                 spec_col_name=spec_col_nameC, do_dummy=False)
    cont_loc = location_dat_process(pdlls, one_hot_col_name=dummy_col_nameL,cont_col_name=cont_col_nameL, \
                                    do_dummy=False)

    XC_comp = apply_para_normalize_dat(cont_comp[cont_col_nameC], comp_norm_param['C_comp'],
                                              comp_norm_param['S_comp'])
    XC_loc = apply_para_normalize_dat(cont_loc['data'][cont_col_nameL], loc_norm_param['C_loc'],
                                             loc_norm_param['S_loc'])

    Y_loc = pdlls[key_col_loc].to_numpy()
    Y_comp = pdccs[key_col_comp].to_numpy()

    X_comp = np.concatenate([Y_comp, XC_comp, XD_comp], axis=1)
    X_loc = np.concatenate([Y_loc, XC_loc, XD_loc], axis=1)

    y_comp_name = key_col_comp
    y_loc_name = key_col_loc
    c_comp_name = cont_col_nameC
    d_comp_name = dummy_col_nameC
    c_loc_name = cont_col_nameL
    d_loc_name = dummy_col_nameL


    dat_comp_pd = pd.DataFrame(data=X_comp, columns=y_comp_name + c_comp_name + d_comp_name)
    dat_loc_pd = pd.DataFrame(data=X_loc, columns=y_loc_name + c_loc_name + d_loc_name)

    print(dat_comp_pd.to_numpy().mean())
    print(dat_loc_pd.to_numpy()[:, 1:].mean())
    print(dat_comp_pd.shape)

    print('Done')

    # print('Final merge...')
    train_test_val_pair = pd.concat(train_test_val_pairs)

    train_test_val_pair.to_csv(pjoin(datapath, 'train_val_test_location_company_82split' + appsadd))
    dat_comp_pd.to_csv(pjoin(datapath, 'company_feat' + appsadd))
    dat_loc_pd.to_csv(pjoin(datapath, 'location_feat' + appsadd))
    print('All Done')

    print(dat_comp_pd.shape,dat_loc_pd.shape)