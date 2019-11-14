import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import pandas as pd
import numpy as np
import pygeohash as pgh
import argparse
from math import *

pjoin = os.path.join

# city filter
def cityfilter(datComp, datLoc):
    city = datComp.groupby(['physical_city'], as_index=False)['physical_city'].agg({'cnt': 'count'})
    print(len(city))
    pdatLoc = pd.merge(datLoc, city, how='inner', left_on=['city'], right_on=['physical_city'],
                       suffixes=['_loc', '_comp']).reset_index(drop=True)
    return pdatLoc


def geohash(data, precision=6):
    data['geohash'] = data.apply(lambda row: pgh.encode(row['longitude'], row['latitude'], precision=precision), axis=1)


def geo_distance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    dis = 2 * asin(sqrt(a)) * 6371 * 1000
    return dis


def duplicateCheck(data, colname: str) -> bool:
    R = data.groupby([colname], as_index=False)[colname].agg({'cnt': 'count'})
    R = R[R['cnt'] > 1]
    if len(R) > 0:
        print('duplicate detected')
        return False
    else:
        print('not duplicate')
        return True


def calcLinkTable(datComp, datLoc, verbose=True):
    if not verbose:
        print('merging...')
    df_cartesian = pd.merge(datComp, datLoc, on='geohash', how='outer', suffixes=['_comp', '_loc'])
    if not verbose:
        print(list(df_cartesian.columns))
        print(len(df_cartesian))
        print('calc geo dist...')
    df_cartesian['geo_distance'] = df_cartesian.apply(
        lambda row: geo_distance(row['longitude_comp'], row['latitude_comp'], row['longitude_loc'],
                                 row['latitude_loc']), axis=1)
    if not verbose:
        print('sort geo dist')
    df_cartesian_min_distance = df_cartesian.sort_values(by="geo_distance").groupby(["duns_number"],
                                                                                    as_index=False).first()

    result = df_cartesian_min_distance[
        ['duns_number', 'atlas_location_uuid', 'geo_distance', 'longitude_loc', 'latitude_loc']]
    if not verbose:
        duplicateCheck(result, 'atlas_location_uuid')
    return result


def calcLinkTablev2(datComp, datLoc, dist_thresh=500,verbose=True):
    if not verbose:
        print('merging...')
    df_cartesian = pd.merge(datLoc, datComp, on='geohash', how='outer', suffixes=['_loc', '_comp'])
    if not verbose:
        print(list(df_cartesian.columns))
        print(len(df_cartesian))
        print('calc geo dist...')

    if dist_thresh > 0:
        df_cartesian['geo_distance'] = df_cartesian.apply(
            lambda row: geo_distance(row['longitude_comp'], row['latitude_comp'], row['longitude_loc'],
                                     row['latitude_loc']), axis=1)
        df_loc_comp = df_cartesian[df_cartesian['geo_distance'] <= dist_thresh]

    else:
        df_loc_com = df_cartesian

    result = df_loc_comp[['duns_number', 'atlas_location_uuid', 'longitude_loc', 'latitude_loc']]
    num_used_loc = len(result.groupby('atlas_location_uuid').first().reset_index())
    num_used_comp = len(result.groupby('duns_number').first().reset_index())
    num_loc = len(datLoc)
    num_comp = len(datComp)
    print('location used: %d, location total: %d, coverage:%0.3f' % (num_used_loc, num_loc, num_used_loc / num_loc))
    print('company used: %d, company total: %d, coverage:%0.3f' % (num_used_comp, num_comp, num_used_comp / num_comp))
    return result


def fuzzy_geosearchv2(datComp, datLoc, precision=5, thresh=500):
    print('Initial company num:', len(datComp))
    datLoc_city = cityfilter(datComp, datLoc)
    print(len(datComp), len(datLoc_city))
    datComp_city = datComp[['duns_number', 'longitude', 'latitude']]
    datLoc_city = datLoc_city[['atlas_location_uuid', 'longitude', 'latitude']]

    geohash(datComp_city, precision)
    geohash(datLoc_city, precision)
    linkCL = calcLinkTablev2(datComp_city, datLoc_city, dist_thresh=thresh)

    return linkCL

if __name__ == '__main__':
    # data load
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--run_root', default='/Users/yefeichen/Database/location_recommender_system/')
    arg('--ls_card',default='location_scorecard_191113.csv')
    arg('--apps',default='_191114.csv')
    arg('--geo_bit',type=int,default=6)
    arg('--dist_thresh',type=float,default=500)
    args = parser.parse_args()


    datapath =args.run_root
    cfile = ['dnb_pa.csv', 'dnb_sf.csv', 'dnb_sj.csv', 'dnb_Los_Angeles.csv', 'dnb_New_York.csv']
    lfile = args.ls_card
    apps = args.apps
    precision = args.geo_bit
    dist_thresh = args.dist_thresh

    print(datapath,lfile,apps,precision,dist_thresh)

    pdc1 = pd.read_csv(pjoin(datapath, cfile[0]))
    pdc2 = pd.read_csv(pjoin(datapath, cfile[1]))
    pdc3 = pd.read_csv(pjoin(datapath, cfile[2]))
    pdc4 = pd.read_csv(pjoin(datapath, cfile[3]))
    pdc5 = pd.read_csv(pjoin(datapath, cfile[4]))

    # pdc = pd.concat([pdc1,pdc2,pdc3],axis=0)
    pdl = pd.read_csv(pjoin(datapath, lfile))

    linkCL1 = fuzzy_geosearchv2(pdc1,pdl,precision=precision,thresh=dist_thresh)
    print(len(linkCL1))
    linkCL1.to_csv(pjoin(datapath,'PA'+apps),index = None, header=True)
    del linkCL1

    linkCL2 = fuzzy_geosearchv2(pdc2,pdl,precision=precision,thresh=dist_thresh)
    print(len(linkCL2))
    linkCL2.to_csv(pjoin(datapath,'SF'+apps),index = None, header=True)
    del linkCL2

    linkCL3 = fuzzy_geosearchv2(pdc3,pdl,precision=precision,thresh=dist_thresh)
    print(len(linkCL3))
    linkCL3.to_csv(pjoin(datapath,'SJ'+apps),index = None, header=True)
    del linkCL3

    linkCL4 = fuzzy_geosearchv2(pdc4,pdl,precision=precision,thresh=dist_thresh)
    print(len(linkCL4))
    linkCL4.to_csv(pjoin(datapath,'LA'+apps),index = None, header=True)
    del linkCL4

    linkCL5 = fuzzy_geosearchv2(pdc5,pdl,precision=precision,thresh=dist_thresh)
    print(len(linkCL5))
    linkCL5.to_csv(pjoin(datapath,'NY'+apps),index = None, header=True)
    del linkCL5

