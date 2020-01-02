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
    args = parser.parse_args()

    datapath = args.run_root
    cfile = origin_comp_file
    app_date = args.app_date
    apps = app_date + '.csv'
    lfile = args.ls_card  # It is fixed as input
    clfile = [c + apps for c in cityabbr]

    print('Args:',datapath,apps,lfile,args.ratio)

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

    dat_comp_pds = []
    dat_loc_pds = []

    pdlls = []  # all location feat pd list
    pdccs = []
    for ind_city in range(5):
        print('processing city: %s'% citylongname[ind_city])
        pdc = pd.read_csv(pjoin(datapath, cfile[ind_city]))
        pdl = pd.read_csv(pjoin(datapath, lfile))
        pdcl = pd.read_csv(pjoin(datapath, clfile[ind_city]))

        # building features
        col_list = list(pdl.columns)
        pdll = pdl.merge(pdcl, how='inner', on=['atlas_location_uuid'], suffixes=['', '_right'])
        pdll = pdll.groupby(['atlas_location_uuid']).first().reset_index()
        pdll = pdll[col_list]
        pdlls.append(pdll)

        # company feature
        pdc['city'] = ind_city
        pdccs.append(pdc)

    # for loop end
    pdlls = pd.concat(pdlls, axis=0)
    pdccs = pd.concat(pdccs, axis=0)

    # building feature
    pdlls = pdlls.reset_index()
    proc_pdl = location_dat_process(pdlls,one_hot_col_name=dummy_col_nameL,cont_col_name=cont_col_nameL)

    # company feature
    pdccs = pdccs.reset_index()
    proc_pdc = comp_dat_process(pdccs,one_hot_col_name=dummy_col_nameC,cont_col_name=cont_col_nameC,spec_col_name=spec_col_nameC)
    print(len(proc_pdc))

    print('start saving company and location feature...')

    XC_comp, XD_comp, Y_comp, c_comp_name, d_comp_name, y_comp_name = transpd2np_single(proc_pdc, cont_col_nameC,
                                                                                        not_feat_col,
                                                                                        id_col_name=key_col_comp)
    XC_loc, XD_loc, Y_loc, c_loc_name, d_loc_name, y_loc_name = transpd2np_single(proc_pdl['data'], cont_col_nameL,
                                                                                  not_feat_col, id_col_name=key_col_loc)

    # one hot explanation
    comp_one_hot_col_name = dummy_col_nameC #['major_industry_category', 'location_type', 'primary_sic_2_digit']
    loc_one_hot_col_name = dummy_col_nameL #['building_class']

    loc_coldict = {}
    for colname in loc_one_hot_col_name:
        loc_coldict[colname] = []
        for dummyname in d_loc_name:
            if dummyname.startswith(colname):
                catname = dummyname.replace(colname, '', 1)  # replace only once(for the sake of protection)
                loc_coldict[colname].append(catname[1:])

    comp_coldict = {}
    for colname in comp_one_hot_col_name:
        comp_coldict[colname] = []
        for dummyname in d_comp_name:
            if dummyname.startswith(colname):
                catname = dummyname.replace(colname, '', 1)  # replace only once(for the sake of protection)
                comp_coldict[colname].append(catname[1:])

    print(comp_coldict, loc_coldict)

    save_obj(comp_coldict, pjoin(datapath, 'comp_feat_dummy_param' + app_date))
    save_obj(loc_coldict, pjoin(datapath, 'loc_feat_dummy_param' + app_date))

    print('one hot description stored...')

    print('Start normalization')

    C_comp, S_comp = get_para_normalize_dat(XC_comp)
    C_loc, S_loc = get_para_normalize_dat(XC_loc)
    XC_comp = apply_para_normalize_dat(XC_comp, C_comp, S_comp)
    XC_loc = apply_para_normalize_dat(XC_loc, C_loc, S_loc)

    X_comp = np.concatenate([Y_comp, XC_comp, XD_comp], axis=1)
    X_loc = np.concatenate([Y_loc, XC_loc, XD_loc], axis=1)

    comp_norm_param = {
        'C_comp': C_comp,
        'S_comp': S_comp,
        'columns': c_comp_name
    }

    loc_norm_param = {
        'C_loc': C_loc,
        'S_loc': S_loc,
        'columns': c_loc_name
    }

    save_obj(comp_norm_param, pjoin(datapath, 'comp_feat_norm_param' + app_date))
    save_obj(loc_norm_param, pjoin(datapath, 'loc_feat_norm_param' + app_date))

    dat_comp_pd = pd.DataFrame(data=X_comp, columns=y_comp_name + c_comp_name + d_comp_name)
    dat_comp_pd = pd.concat([dat_comp_pd,proc_pdc[['city']]], axis=1)


    dat_loc_pd = pd.DataFrame(data=X_loc, columns=y_loc_name + c_loc_name + d_loc_name)

    print(dat_comp_pd.to_numpy().mean())
    print(dat_loc_pd.to_numpy()[:, 1:].mean())
    print(dat_comp_pd.shape)

    print('Done')

    dat_comp_pd.to_csv(pjoin(datapath, 'company_feat' + apps))
    dat_loc_pd.to_csv(pjoin(datapath, 'location_feat' + apps))
    print('All Done')

    print(dat_comp_pd.shape,dat_loc_pd.shape)