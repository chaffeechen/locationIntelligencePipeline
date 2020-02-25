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
    df_cartesian = pd.merge(datLoc, datComp, on='geohash', how='left', suffixes=['_loc', '_comp'])
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
        df_loc_comp = df_cartesian

    result = df_loc_comp[['duns_number', 'atlas_location_uuid', 'longitude_loc', 'latitude_loc']]
    num_used_loc = len(result.groupby('atlas_location_uuid').first().reset_index())
    num_used_comp = len(result.groupby('duns_number').first().reset_index())
    num_loc = len(datLoc)
    num_comp = len(datComp)
    print('location used: %d, location total: %d, coverage:%0.3f' % (num_used_loc, num_loc, num_used_loc / num_loc))
    print('company used: %d, company total: %d, coverage:%0.3f' % (num_used_comp, num_comp, num_used_comp / num_comp))
    return result


def fuzzy_geosearchv2(datComp, datLoc, precision=5, thresh=500):
    """
    Each company can have multiply location in range controlled by precision and threshold.
    :param datComp: 
    :param datLoc: 
    :param precision: 
    :param thresh: 
    :return: 
    """
    print('Initial company num:', len(datComp))
    datLoc_city = cityfilter(datComp, datLoc)
    print(len(datComp), len(datLoc_city))
    datComp_city = datComp[['duns_number', 'longitude', 'latitude']]
    datLoc_city = datLoc_city[['atlas_location_uuid', 'longitude', 'latitude']]

    geohash(datComp_city, precision)
    geohash(datLoc_city, precision)
    linkCL = calcLinkTablev2(datComp_city, datLoc_city, dist_thresh=thresh)

    return linkCL


def fuzzy_geosearch(datComp, datLoc, precision=[8, 7, 6, 5], thresh=[500, 1000, 1000, 1000]):
    """
    A coarse to fine way to assign each company a unique location. 
    :param datComp: 
    :param datLoc: 
    :param precision: 
    :param thresh: 
    :return: 
    """
    print('Initial company num:', len(datComp))
    datLoc_city = cityfilter(datComp, datLoc)
    print(len(datComp), len(datLoc_city))
    datComp_city = datComp[['duns_number', 'longitude', 'latitude']]
    datLoc_city = datLoc_city[['atlas_location_uuid', 'longitude', 'latitude']]
    datlist = []

    for i, p in enumerate(precision):
        print('level:', p)
        geohash(datComp_city, p)
        geohash(datLoc_city, p)
        linkCL = calcLinkTable(datComp_city, datLoc_city)
        datlist.append(linkCL[linkCL['geo_distance'] <= thresh[i]])
        unmatched = linkCL[linkCL['geo_distance'] > thresh[i]].groupby('duns_number', as_index=False).first()
        if len(unmatched) == 0:
            print('all companies matched with a building')
            break
        datComp_city = pd.merge(datComp_city, unmatched['duns_number'], on='duns_number', how='inner')
        print('datComp_city:', len(datComp_city))

    res = pd.concat(datlist, axis=0, ignore_index=True)
    print('Initial company num:', len(datComp), 'vs. Remain company num:', len(res), 'rate:=',
          float(len(res)) / len(datComp))
    return res

if __name__ == '__main__':
    # data load
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--run_root', default='/Users/yefeichen/Database/location_recommender_system/')
    arg('--ls_card',default='location_scorecard_191113.csv')
    arg('--apps',default='_191114.csv')
    arg('--geo_bit',type=int,default=6)
    arg('--dist_thresh',type=float,default=500)
    arg('--single',action='store_true',help='each company has a unique location')
    args = parser.parse_args()


    datapath =args.run_root
    cfile = ['dnb_pa.csv', 'dnb_sf.csv', 'dnb_sj.csv', 'dnb_Los_Angeles.csv', 'dnb_New_York.csv']
    clfile = ['PA', 'SF', 'SJ', 'LA', 'NY']


    lfile = args.ls_card
    apps = args.apps
    precision = args.geo_bit
    dist_thresh = args.dist_thresh

    print(datapath,lfile,apps,precision,dist_thresh)

    clfile = [c + apps for c in clfile]

    pdl = pd.read_csv(pjoin(datapath, lfile))

    #shrink into 1 code
    for ind_city in range(5):
        pdc = pd.read_csv(pjoin(datapath, cfile[ind_city]))
        if args.single:
            linkCL = fuzzy_geosearch(pdc, pdl, precision=[8, 7, 6], thresh=[500, 1000, 1000])
        else:
            linkCL = fuzzy_geosearchv2(pdc,pdl,precision=precision,thresh=dist_thresh)
        print(len(linkCL))
        linkCL.to_csv(pjoin(datapath,clfile[ind_city]),index = None, header=True)


