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

pjoin = os.path.join


# function_base
def getPosNegdat(dat):
    """
    dat: pos pair of data (location,company,geo,distance)
    return pos/neg pair of data, same structure of dat except one more column for label
    """
    shuffle_dat = dat.sample(frac=1).reset_index(drop=True)

    # shuffle_dat.head()

    twin_dat = dat.join(shuffle_dat, how='left', lsuffix='_left', rsuffix='_right')
    twin_dat = twin_dat[twin_dat['atlas_location_uuid_left'] != twin_dat['atlas_location_uuid_right']]
    print(len(twin_dat))
    twin_dat.head()

    neg_datA = twin_dat[['duns_number_left', 'atlas_location_uuid_right', 'longitude_loc_right', 'latitude_loc_right']]
    neg_datA = neg_datA.rename(
        columns={'duns_number_left': 'duns_number', 'atlas_location_uuid_right': 'atlas_location_uuid',
                 'longitude_loc_right': 'longitude_loc', 'latitude_loc_right': 'latitude_loc'})

    neg_datB = twin_dat[['duns_number_right', 'atlas_location_uuid_left', 'longitude_loc_left', 'latitude_loc_left']]
    neg_datB = neg_datB.rename(
        columns={'duns_number_right': 'duns_number', 'atlas_location_uuid_left': 'atlas_location_uuid',
                 'longitude_loc_left': 'longitude_loc', 'latitude_loc_left': 'latitude_loc'})

    neg_dat = pd.concat([neg_datA, neg_datB], axis=0)
    neg_dat['label'] = 0
    dat['label'] = 1
    res_dat = pd.concat(
        [dat[['duns_number', 'atlas_location_uuid', 'longitude_loc', 'latitude_loc', 'label']], neg_dat], axis=0)
    print('Neg dat num:', len(neg_dat), ';Pos dat num:', len(dat))
    return res_dat


def getPosNegdatv2_fast(dat):
    """
    dat: pos pair of data (location,company,geo,distance)
    return pos/neg pair of data, same structure of dat except one more column for label
    """
    shuffle_dat = dat.sample(frac=1).reset_index(drop=True)
    twin_dat = dat.join(shuffle_dat, how='left', lsuffix='_left', rsuffix='_right')

    pot_neg_datA = twin_dat[
        ['duns_number_left', 'atlas_location_uuid_right', 'longitude_loc_right', 'latitude_loc_right']] \
        .rename(columns={'duns_number_left': 'duns_number', 'atlas_location_uuid_right': 'atlas_location_uuid',
                         'longitude_loc_right': 'longitude_loc', 'latitude_loc_right': 'latitude_loc'})

    pot_neg_datB = twin_dat[
        ['duns_number_right', 'atlas_location_uuid_left', 'longitude_loc_left', 'latitude_loc_left']] \
        .rename(columns={'duns_number_right': 'duns_number', 'atlas_location_uuid_left': 'atlas_location_uuid',
                         'longitude_loc_left': 'longitude_loc', 'latitude_loc_left': 'latitude_loc'})

    pot_neg_dat = pd.concat([pot_neg_datA, pot_neg_datB], axis=0)
    pot_neg_dat['label'] = 0
    dat['label'] = 1

    # col alignment
    col_list = ['duns_number', 'atlas_location_uuid', 'label']
    dat = dat[col_list]
    pot_neg_dat = pot_neg_dat[col_list]

    # clean pos dat in neg dat
    neg_dat = pd.merge(pot_neg_dat, dat, on=['duns_number', 'atlas_location_uuid'], how='left',
                       suffixes=['', '_right']).reset_index(drop=True)
    neg_dat['label'] = neg_dat['label'].fillna(0)
    neg_dat = neg_dat[neg_dat['label_right'] != 1]

    print('Clean %d neg data into %d real neg data.' % (len(pot_neg_dat), len(neg_dat)))

    res_dat = pd.concat([dat, neg_dat], axis=0).reset_index(drop=True)
    # res_dat = res_dat.groupby(['duns_number', 'atlas_location_uuid'])['label'].max().reset_index()
    return res_dat


def getPosNegdatv2(dat):
    """
    dat: pos pair of data (location,company,geo,distance)
    return pos/neg pair of data, same structure of dat except one more column for label
    """
    shuffle_dat = dat.sample(frac=1).reset_index(drop=True)

    # shuffle_dat.head()

    twin_dat = dat.join(shuffle_dat, how='left', lsuffix='_left', rsuffix='_right')

    pot_neg_datA = twin_dat[
        ['duns_number_left', 'atlas_location_uuid_right', 'longitude_loc_right', 'latitude_loc_right']] \
        .rename(columns={'duns_number_left': 'duns_number', 'atlas_location_uuid_right': 'atlas_location_uuid',
                         'longitude_loc_right': 'longitude_loc', 'latitude_loc_right': 'latitude_loc'})

    pot_neg_datB = twin_dat[
        ['duns_number_right', 'atlas_location_uuid_left', 'longitude_loc_left', 'latitude_loc_left']] \
        .rename(columns={'duns_number_right': 'duns_number', 'atlas_location_uuid_left': 'atlas_location_uuid',
                         'longitude_loc_left': 'longitude_loc', 'latitude_loc_left': 'latitude_loc'})

    pot_neg_dat = pd.concat([pot_neg_datA, pot_neg_datB], axis=0)
    pot_neg_dat['label'] = 0
    dat['label'] = 1
    # col alignment
    col_list = ['duns_number', 'atlas_location_uuid', 'label']
    dat = dat[col_list]
    pot_neg_dat = pot_neg_dat[col_list]
    res_dat = pd.concat([dat, pot_neg_dat], axis=0)
    res_dat = res_dat.groupby(['duns_number', 'atlas_location_uuid'])['label'].max().reset_index()
    return res_dat


def splitdat(dat, key_column=['duns_number'], right_colunm='atlas_location_uuid_tr', rate_tr=0.8):
    """
    split the <company,location> pair into training/testing dat
    """
    tr = dat.sample(frac=rate_tr)
    tt = pd.merge(dat, tr, on=key_column, how='left', suffixes=['', '_tr'])
    tt = tt[tt[right_colunm].isnull()]
    tt = tt[list(tr.columns)]
    print('Train dat:', len(tr), 'Test dat:', len(tt))
    return tr, tt


# data process
def onehotdat(dat, key_column: list, dummy_na=True):
    dat[key_column] = dat[key_column].astype(str)
    dum_dat = pd.get_dummies(dat[key_column], dummy_na=dummy_na)  # it has nan itself
    return dum_dat


def split2num(emp_range: str):
    max_emp_val = emp_range.replace(' ', '').split('-')
    if len(max_emp_val) < 2:
        return 10
    else:
        return float(max_emp_val[1])


def max_col(dat, col, minval=1):
    dat[col] = dat[col].apply(lambda r: max(r, minval))


def comp_dat_process(dat, one_hot_col_name, cont_col_name , spec_col_name ,do_dummy=True):
    """
    pd -> company key,cont_feature,spec_feature,dum_feature
    """
    # one_hot_col_name = ['major_industry_category', 'location_type', 'primary_sic_2_digit']
    # spec_col_name = 'emp_here_range'
    # cont_col_name = ['emp_here', 'emp_total', 'sales_volume_us', 'square_footage']
    cont_col_name = [ c for c in cont_col_name if c not in spec_col_name ]

    if do_dummy:
        print('doing one-hot...')
        dum_dat = onehotdat(dat, one_hot_col_name)

    print('extract continuous...')
    cont_dat = dat[cont_col_name].fillna(value=0).astype(float)

    print('specific feature')
    spec_dat = dat[spec_col_name].fillna(value='1-10').astype(str)
    spec_dat = spec_dat.apply(lambda row: split2num(row))

    max_col(cont_dat, 'emp_here', 1)

    if do_dummy:
        res_dat = dat[['duns_number']].join([cont_dat, spec_dat, dum_dat], how='left')
    else:
        res_dat = dat[['duns_number']].join([cont_dat, spec_dat], how='left')

    if do_dummy:
        assert (len(res_dat) == len(dum_dat))
    assert (len(res_dat) == len(cont_dat))
    assert (len(res_dat) == len(spec_dat))
    return res_dat


def location_dat_process(dat, one_hot_col_name, cont_col_name ,do_dummy=True):
    """
    pd -> location key,cont_feature,dum_feature
    """
    # one_hot_col_name = ['building_class']
    # cont_col_name = ['score_predicted_eo', 'score_employer', 'num_emp_weworkcore', 'num_poi_weworkcore',
    #                  'pct_wwcore_employee', 'pct_wwcore_business', 'num_retail_stores', 'num_doctor_offices',
    #                  'num_eating_places', 'num_drinking_places', 'num_hotels', 'num_fitness_gyms',
    #                  'population_density', 'pct_female_population', 'median_age', 'income_per_capita',
    #                  'pct_masters_degree', 'walk_score', 'bike_score']
    if do_dummy:
        print('doing one-hot...')
        dum_dat = onehotdat(dat, one_hot_col_name, False)
        print(len(dum_dat))

    print('extract continuous...')
    cont_dat = dat[cont_col_name].fillna(value=0).astype(float)
    print(len(cont_dat))

    if do_dummy:
        res_dat = dat[['atlas_location_uuid']].join([cont_dat, dum_dat], how='left')
    else:
        res_dat = dat[['atlas_location_uuid']].join([cont_dat], how='left')
    print(len(res_dat))

    if do_dummy:
        assert (len(res_dat) == len(dum_dat))
    assert (len(res_dat) == len(cont_dat))

    if do_dummy:
        return {'data': res_dat,
                'cont_feat_num': len(list(cont_dat.columns)),
                'dum_feat_num': len(list(dum_dat.columns))}
    else:
        return {'data': res_dat,
                'cont_feat_num': len(list(cont_dat.columns))}


def normalize_dat_v2(trX, ttX, axis=0):
    center = trX.mean(axis=axis)
    center = np.expand_dims(center, axis)
    scale = trX.std(axis=axis)
    scale = np.expand_dims(scale, axis)

    trX = (trX - center) / scale
    ttX = (ttX - center) / scale
    return trX, ttX


def get_para_normalize_dat(trX, axis=0):
    center = trX.mean(axis=axis)
    scale = trX.std(axis=axis)
    scale += 1e-4
    return center, scale


def apply_para_normalize_dat(X, center, scale, axis=0):
    """
    X can be pd or numpy!
    """
    center = np.expand_dims(center, axis)
    scale = np.expand_dims(scale, axis)
    X = (X - center) / scale
    return X

def normalize_dat(trX, ttX, cols=5, axis=0):
    D = trX[:, :cols]
    center = D.mean(axis=axis)
    center = np.expand_dims(center, axis)
    scale = D.std(axis=axis)
    scale = np.expand_dims(scale, axis)

    trX[:, :cols] = (D - center) / scale
    ttX[:, :cols] = (ttX[:, :cols] - center) / scale

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def transpd2np_single(featdat,cont_col_name:list,not_feat_col:list,id_col_name:list):
    XC = featdat.loc[:,cont_col_name].to_numpy()
    out_col = not_feat_col+cont_col_name
    dum_col_name = [col for col in list(featdat.columns) if col not in out_col]
    XD = featdat.loc[:,dum_col_name].to_numpy()
    Y = featdat[id_col_name].to_numpy()
    return XC,XD,Y,cont_col_name,dum_col_name,id_col_name



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--run_root', default='/Users/yefeichen/Database/location_recommender_system/')
    arg('--ls_card',default='location_scorecard_191113.csv')
    arg('--app_date',default='_191114')
    arg('--ratio',type=float,default=0.8)
    args = parser.parse_args()

    datapath = args.run_root
    cfile = ['dnb_pa.csv', 'dnb_sf.csv', 'dnb_sj.csv', 'dnb_Los_Angeles.csv', 'dnb_New_York.csv']
    app_date = args.app_date
    apps = app_date + '.csv'
    lfile = args.ls_card  # It is fixed as input
    clfile = ['PA', 'SF', 'SJ', 'LA', 'NY']
    clfile = [c + apps for c in clfile]

    print('Args:',datapath,apps,lfile,args.ratio)

    not_feat_col = ['duns_number',
                    'atlas_location_uuid',
                    'longitude_loc',
                    'latitude_loc',
                    'label']
    cont_col_nameC = ['emp_here', 'emp_total', 'sales_volume_us', 'square_footage', 'emp_here_range']
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
    for ind_city in range(5):
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
    comp_one_hot_col_name = ['major_industry_category', 'location_type', 'primary_sic_2_digit']
    loc_one_hot_col_name = ['building_class']

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
    dat_loc_pd = pd.DataFrame(data=X_loc, columns=y_loc_name + c_loc_name + d_loc_name)

    print(dat_comp_pd.to_numpy().mean())
    print(dat_loc_pd.to_numpy()[:, 1:].mean())
    print(dat_comp_pd.shape)

    print('Done')

    # print('Final merge...')
    train_test_val_pair = pd.concat(train_test_val_pairs)

    train_test_val_pair.to_csv(pjoin(datapath, 'train_val_test_location_company_82split' + apps))
    dat_comp_pd.to_csv(pjoin(datapath, 'company_feat' + apps))
    dat_loc_pd.to_csv(pjoin(datapath, 'location_feat' + apps))
    print('All Done')

    print(dat_comp_pd.shape,dat_loc_pd.shape)