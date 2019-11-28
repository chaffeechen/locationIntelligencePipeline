import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder
from enum import Enum

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
    cont_col_name = [ c for c in cont_col_name if c != spec_col_name ]

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
        res_dat = dat[['duns_number','city']].join([cont_dat, spec_dat, dum_dat], how='left')
    else:
        res_dat = dat[['duns_number','city']].join([cont_dat, spec_dat], how='left')

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

def apply_dummy(coldict:dict,data):
    cat_list = []
    dummy_num = len(coldict)
    dummy_name = []
    for key in coldict:
        cat_list.append(np.array(coldict[key])) #list of array for onehot engine
        dummy_name = dummy_name + [ key+'_'+col for col in coldict[key] ] #full name of dummy col name

    enc = OneHotEncoder(handle_unknown='ignore',categories=cat_list)

    origin_dummy_col = [key for key in coldict]
    result = enc.fit_transform(data[origin_dummy_col]).toarray()
    #array to pd
    pd_new = pd.DataFrame(data=result,columns=dummy_name)
    return pd_new

#=======================================================================================================================
#=======================================================================================================================
# get_sub_recommend_reason_after_similarity
#=======================================================================================================================
#=======================================================================================================================
def generate_loc_type(comp_feat, comp_loc, matching_col):
    # matching_col = 'major_industry_category'
    comp_type = comp_feat[['duns_number', matching_col]]
    comp_type_location = pd.merge(comp_type, comp_loc[['duns_number', 'atlas_location_uuid']], on='duns_number',
                                  how='inner')

    loc_type = comp_type_location.groupby(['atlas_location_uuid', matching_col]).first().reset_index()[
        ['atlas_location_uuid', matching_col]]
    return loc_type


class sub_rec_similar_company(object):
    def __init__(self, comp_feat, comp_loc, matching_col,reason_col_name='reason'):
        """
        comp_feat: original company information
        comp_loc: company-location affinities of a certain city
        matching_col = 'major_industry_category' big category
                    or 'primary_sic_2_digit' more detailed category
        """
        self.comp_feat = comp_feat
        self.comp_loc = comp_loc
        self.matching_col = matching_col
        self.reason_col_name = reason_col_name
        self.loc_type = generate_loc_type(comp_feat, comp_loc, matching_col)

    def get_candidate_location_for_company(self, query_comp_feat,reason='similar company inside'):
        sub_pairs = pd.merge(query_comp_feat[['duns_number', self.matching_col]], self.loc_type, on=self.matching_col,
                             how='left', suffixes=['', '_right'])
        sub_pairs = sub_pairs[sub_pairs['atlas_location_uuid'].notnull()]#sometimes a company may have no location to recommend
        sub_pairs[self.reason_col_name] = reason
        return sub_pairs


class global_filter(object):
    def __init__(self, loc_feat):
        self.loc_feat = loc_feat

    def filtering(self, key_column, percentile=0.2, mode='gt'):
        val = self.loc_feat[[key_column]].quantile(q=percentile).item()
        if mode == 'gt':
            sub_loc = self.loc_feat[self.loc_feat[key_column] >= val]
        else:
            sub_loc = self.loc_feat[self.loc_feat[key_column] <= val]

        self.loc_feat = sub_loc.reset_index(drop=True)
        return self

    def city_filter(self, city_name, key_column='city'):
        self.loc_feat = self.loc_feat[self.loc_feat[key_column] == city_name].reset_index(drop=True)
        return self

    def exfiltering(self, loc_feat, key_column, percentile=0.2, mode='gt'):
        val = loc_feat[[key_column]].quantile(q=percentile).item()
        if mode == 'gt':
            sub_loc = self.loc_feat[self.loc_feat[key_column] >= val]
        else:
            sub_loc = self.loc_feat[self.loc_feat[key_column] <= val]

        return sub_loc.reset_index(drop=True)

    def end(self):
        return self.loc_feat


class sub_rec_condition(object):
    def __init__(self, loc_feat):
        """
        comp_loc: company-location affinities of a certain city
        cond_col = column of location used for filtering
        """
        self.loc_feat = loc_feat
        self.cond_col = []
        self.reason = []

    def filtering(self, cond_col, percentile=0.5, reason='many things'):
        self.cond_col.append(cond_col)
        val = self.loc_feat[[cond_col]].quantile(q=percentile).item()
        if max(val, 10):
            self.loc_feat = self.loc_feat[self.loc_feat[cond_col] >= val].reset_index(drop=True)
            self.reason.append(reason)
        return self

    def exfiltering(self, cond_col, percentile=0.6, reason='many things',reason_col_name='reason'):
        self.cond_col.append(cond_col)
        val = self.loc_feat[[cond_col]].quantile(q=percentile).item()
        if max(val, 10):
            sub_loc = self.loc_feat[self.loc_feat[cond_col] >= val].reset_index(drop=True)
        sub_loc[reason_col_name] = reason
        return sub_loc[['atlas_location_uuid', reason_col_name]]

    def end(self):
        return self.loc_feat

#======================================================================================================================
def ab(df):
    return ','.join(df.values)


def merge_rec_reason_rowise(sub_pairs, group_cols: list, merge_col: str):
    return sub_pairs.groupby(group_cols)[merge_col].apply(ab).reset_index()


def merge_rec_reason_colwise(sub_pairs, cols=['reason1', 'reason2'],dst_col = 'reason',sep=','):
    sub_pairs[dst_col] = sub_pairs[cols[0]].str.cat(sub_pairs[cols[1]], sep=sep)
    return sub_pairs
#======================================================================================================================

def list2json(x,sep=','):
    x = str(x)
    k = ''
    ltx = x.split(sep)
    for item in ltx:
        if k != '':
            if item != '':
                k = k + ',' + "\"" + item + "\""
            else:
                pass
        else:
            if item != '':
                k = "\""+item+"\""
            else:
                pass
    k = '['+k+']'
    return k

def reason_json_format(df,col_name:str='reason',sep=','):
    df[col_name] = df[col_name].apply(lambda x: '{\"reasons\":' + list2json(x,sep) + '}')
    return df

#======================================================================================================================
#======================================================================================================================
# get_dl_sub_recommend_reason
#======================================================================================================================
#======================================================================================================================
class featsrc(Enum):
    company = 0
    location = 1
    region = 2


class feature_translate(object):
    def __init__(self):
        self.col2phs = {}
        self.init_dict()

    def init_dict(self):
        # company
        self.col2phs['emp_here'] = (featsrc.company, 'local employee number')
        self.col2phs['emp_here_range'] = (featsrc.company, 'local employee number')
        self.col2phs['emp_total'] = (featsrc.company, 'total employee number')
        self.col2phs['sales_volume_us'] = (featsrc.company, 'sales volume')
        self.col2phs['location_type'] = (featsrc.company, 'type(Single,Branch,HQ)')
        self.col2phs['square_footage'] = (featsrc.company, 'expected square footage')
        self.col2phs['primary_sic_2'] = (featsrc.company, 'industry type')
        # building
        self.col2phs['score_predicted_eo'] = (featsrc.location, 'eo')
        self.col2phs['building_class'] = (featsrc.location, 'class')
        # region
        self.col2phs['num_retail_stores'] = (featsrc.region, 'shop amenities')
        self.col2phs['num_doctor_offices'] = (featsrc.region, 'health amenities')
        self.col2phs['num_eating_places'] = (featsrc.region, 'lunch amenities')
        self.col2phs['num_drinking_places'] = (featsrc.region, 'relaxing amenities')
        self.col2phs['num_hotels'] = (featsrc.region, 'trip amenities')
        self.col2phs['num_fitness_gyms'] = (featsrc.region, 'gym amenities')
        self.col2phs['population_density'] = (featsrc.region, 'population')
        self.col2phs['pct_female_population'] = (featsrc.region, 'population structure')
        self.col2phs['median_age'] = (featsrc.region, 'population environment')
        self.col2phs['income_per_capita'] = (featsrc.region, 'income level')
        self.col2phs['pct_masters_degree'] = (featsrc.region, 'education degree')
        self.col2phs['walk_score'] = (featsrc.region, 'walking amenities')
        self.col2phs['bike_score'] = (featsrc.region, 'biking amenities')

    def getItem(self,gvkey):
        #precision matching
        if gvkey in self.col2phs.keys():
            return {'status':True,
                        'key':gvkey,
                        'item':self.col2phs[gvkey]}
        #rough matching
        for key in self.col2phs.keys():
            if gvkey.startswith(key):
                return {'status':True,
                        'key':key,
                        'item':self.col2phs[key]}

        return {'status':False}

    def merge_lst(self, lst: list, pre_phs='', post_phs=''):
        phs = ''
        #         print(lst)
        for c in lst:
            phs = phs + c + ', '
        if lst:
            phs = phs[:-2]  # get rid of last ', '
        # print(phs)
        if pre_phs:
            pre_phs = pre_phs + ' '
        if post_phs:
            post_phs = ' ' + post_phs
        return pre_phs + phs + post_phs

    def merge_phs(self, lst: list):
        phs = ''
        _lst = [p for p in lst if p]
        for p in _lst:
            if phs:
                phs = phs + '; ' + p
            else:
                phs = p
        return phs

    def make_sense(self, input_lst):
        if isinstance(input_lst, list):
            pass
        elif isinstance(input_lst, str):
            input_lst = input_lst.replace('[','',1)
            input_lst = input_lst.replace(']','',1)
            input_lst = [e for e in input_lst.split(',') if e]
        else:
            return 'Err:input type'

        # print(len(input_lst))
        #in case of irrelavant data
        # input_lst = [self.col2phs[key] for key in input_lst if key in self.col2phs.keys()]

        comp_lst, loc_lst, region_lst = [], [], []

        for key in input_lst:
            #             print(key)
            ret = self.getItem(key)

            if ret['status']:
                phss = ret['item']
                if phss[0] == featsrc.company:
                    comp_lst.append(phss[1])
                elif phss[0] == featsrc.location:
                    loc_lst.append(phss[1])
                elif phss[0] == featsrc.region:
                    region_lst.append(phss[1])

                    #         print(comp_lst,loc_lst,region_lst)

        if comp_lst:  # not empty assert
            comp_phs = self.merge_lst(comp_lst, pre_phs='', post_phs='of your company')
        else:
            comp_phs = ''

        if loc_lst:
            loc_phs = self.merge_lst(loc_lst, pre_phs='', post_phs='of the building')
        else:
            loc_phs = ''

        if region_lst:
            region_phs = self.merge_lst(region_lst, pre_phs='', post_phs='of the region')
        else:
            region_phs = ''

        # print(comp_phs,loc_phs,region_phs)
        final_phs = self.merge_phs([comp_phs, loc_phs, region_phs])
        if final_phs:
            return 'According to the ' + final_phs
        else:
            return ''
