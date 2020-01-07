import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import pandas as pd
import argparse
from utils import *

pjoin = os.path.join

sfx = ['', '_right']

from header import *

cid = 'duns_number'
bid = 'atlas_location_uuid'

if __name__ == '__main__':
    """
    Appending reason with score matrix
    """
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--run_root', default='/Users/yefeichen/Database/location_recommender_system/')
    arg('--ls_card', default='location_scorecard_200106.csv')
    arg('--apps', type=str, default='_200106.csv')
    arg('--single',type=int,default=-1)
    arg('--merge', action='store_true', help='doing merge only')
    arg('--nomerge',action='store_true',help='dont merge')
    arg('--sampled', action='store_true', help='is similarity score sampled?')
    arg('--ww', action='store_true', help='is ww building only?')
    arg('--tt', action='store_true', help='doing test for 1 city only')


    args = parser.parse_args()
    testcode = 2  # in --tt mode, which city is used for test
    if args.single >= 0:
        singlecode = min(len(citylongname),args.single)
        print('#####Attention#####')
        print('Only %s will be generated'%citylongname[singlecode])

    datapath = args.run_root
    cfile = origin_comp_file
    lfile = args.ls_card
    clfile = [c + args.apps for c in cityabbr]
    cityname = citylongname

    comp_feat_file = 'company_feat' + args.apps
    comp_feat_normed = pd.read_csv(pjoin(datapath, comp_feat_file), index_col=0)
    loc_feat_file = 'location_feat' + args.apps
    loc_feat_normed = pd.read_csv(pjoin(datapath, loc_feat_file), index_col=0)

    comp_feat_col = [c for c in comp_feat_normed.columns if c not in [cid, bid]]
    loc_feat_col = [c for c in loc_feat_normed.columns if c not in [cid, bid]]

    if args.ww:
        ww = 'ww_'
    else:
        ww = ''

    if args.sampled:
        sam = 'sampled_'
    else:
        sam = 'all_'

    ssfile = [sam + ww + c.replace(args.apps, '') + '_similarity' + args.apps for c in
              clfile]  # e.g. ['ww_PA_similarity_20191113.csv']
    dlsub_ssfile = ['dlsub_' + c for c in ssfile]

    wework_location_only = args.ww

    """
    Reasons cannot be duplicated because they are used as column names.
    (reason_col_name, priority) 
    priority 1 is highest.
    """
    reason_col_name = [
        ('reason_similar_biz', 1),  # sub_pairs
        ('reason_location_based', 6),  # sub_loc_recall
        ('reason_model_based', 7),  # dlsubdat
        ('reason_similar_location', 5),
        ('reason_similar_company', 4),
        ('reason_close_2_current_location', 2),
        ('reason_inventory_bom',3),
    ]

    if args.tt:
        print('#####Attention#####')
        print('This is TEST mode only. Not PROD-DEV.')
    else:
        print('This is PROD-DEV mode.')

    if args.merge:
        print('Skip to merging directly...')
    else:
        for ind_city in range(5):
            if args.single >= 0 and singlecode != ind_city:
                continue
            if args.tt and ind_city != testcode:
                continue
            """
            storage for separate recommendation reasons
            """
            reason_db = {}

            print('##city: %s processing##' % cityname[ind_city])
            comp_feat = pd.read_csv(pjoin(datapath, cfile[ind_city]))
            comp_loc = pd.read_csv(pjoin(datapath, clfile[ind_city]))
            loc_feat = pd.read_csv(pjoin(datapath, lfile))

            # if we only focus on buildings with companies inside
            loc_feat = loc_feat.merge(comp_loc[[bid]].groupby(bid).first().reset_index(),
                                      on=bid, suffixes=sfx)

            print('Global filtering')
            # Global filter: master degree!
            global_ft = global_filter(loc_feat=loc_feat)
            sub_loc_feat = global_ft.city_filter(city_name=cityname[ind_city]).end()

            if wework_location_only:
                sub_loc_feat_ww = sub_loc_feat.loc[sub_loc_feat['is_wework'] == True, :]
                sub_comp_loc = pd.merge(comp_loc, sub_loc_feat_ww[[bid]], on=bid,
                                        suffixes=sfx)  # multi comp loc
            else:
                sub_comp_loc = pd.merge(comp_loc, sub_loc_feat[[bid]], on=bid,
                                        suffixes=sfx)  # multi comp loc

            sspd = pd.read_csv(pjoin(datapath, ssfile[ind_city]), index_col=0)
            print('Loading %d similarity score pairs' % len(sspd))

            print('Begin generating reasons')
            # Reason 1:
            print('1. Is there a company with similar biz inside the location?')
            sub_reason_col_name = reason_col_name[0][0]
            matching_col = 'primary_sic_2_digit_v2'  # matching_col = 'major_industry_category'
            query_comp_loc = sspd[[bid, cid]]
            query_comp_loc = query_comp_loc.merge(comp_feat[[cid, matching_col]], on=cid, suffixes=sfx)

            recall_com1 = sub_rec_similar_company(comp_feat=comp_feat, comp_loc=sub_comp_loc,
                                                  matching_col=matching_col, reason_col_name=sub_reason_col_name,
                                                  bid=bid, cid=cid)

            sub_pairs = recall_com1.get_candidate_location_for_company_fast(query_comp_loc=query_comp_loc, reason='like')
            # explanar
            sub_pairs[
                sub_reason_col_name] = 'This location has a tenant company which is in the same industry as your company.'
            reason_db[sub_reason_col_name] = sub_pairs
            print('==> Total pairs generated: %d' % len(sub_pairs))

            # Reason2:
            print('2. How is region?(Location based reason)')
            sub_reason_col_name = reason_col_name[1][0]
            recall_com2 = sub_rec_condition(sub_loc_feat, bid=bid)
            sub_loc_recall_com2 = recall_com2.exfiltering('num_fitness_gyms', percentile=0.5,
                                                          reason='There are enough gyms to work out',
                                                          reason_col_name=sub_reason_col_name)
            sub_loc_recall_com3 = recall_com2.exfiltering('num_drinking_places', percentile=0.5,
                                                          reason='There are enough bars to have a drink',
                                                          reason_col_name=sub_reason_col_name)
            sub_loc_recall_com4 = recall_com2.exfiltering('num_eating_places', percentile=0.5,
                                                          reason='There are enough restaurants to get food',
                                                          reason_col_name=sub_reason_col_name)
            print('==> %d, %d, %d will be mergeed' % (
                len(sub_loc_recall_com2), len(sub_loc_recall_com3), len(sub_loc_recall_com4)))

            sub_loc_recall = pd.concat([sub_loc_recall_com2, sub_loc_recall_com3, sub_loc_recall_com4], axis=0)

            if wework_location_only:
                sub_loc_recall = sub_loc_recall.merge(sub_loc_feat_ww[[bid]], on=bid,
                                                      suffixes=sfx)
            # explanar:merge_rec_reason_rowise 需要在结尾加"."
            sub_loc_recall = merge_rec_reason_rowise(sub_loc_recall, group_cols=[bid],
                                                     merge_col=sub_reason_col_name, sep='. ')
            sub_loc_recall[sub_reason_col_name] = 'This building is at a location with great amenities: ' + sub_loc_recall[
                sub_reason_col_name] + '. '

            print('sub_loc_recall sized %d' % len(sub_loc_recall))
            reason_db[sub_reason_col_name] = sub_loc_recall

            # Reason3: Tag!!!!
            print('3. Model based Reason(Implicit reason)')
            sub_reason_col_name = reason_col_name[2][0]
            featTranslator = feature_translate()
            dlsubdat = pd.read_csv(pjoin(datapath, dlsub_ssfile[ind_city]), index_col=0)
            dlsubdat[sub_reason_col_name] = dlsubdat.apply(lambda row: featTranslator.make_sense(row['merged_feat']),
                                                           axis=1)
            dlsubdat = dlsubdat[[bid, cid, sub_reason_col_name]]
            print('==> Total pairs generated: %d' % len(dlsubdat))
            reason_db[sub_reason_col_name] = dlsubdat

            # print('similarity score sampled pairs: %d' % len(sample_sspd))

            print('4. Is the recommended location similar with its current one?')
            sub_reason_col_name = reason_col_name[3][0]
            cont_col_nameL = feature_column['cont_col_nameL']
            dummy_col_nameL = feature_column['dummy_col_nameL']
            recall_com4 = sub_rec_similar_location(cont_col_name=cont_col_nameL, dummy_col_name=dummy_col_nameL,
                                                   reason_col_name=sub_reason_col_name, cid=cid, bid=bid)
            loc_comp_loc = recall_com4.get_reason(sspd=sspd, comp_loc=comp_loc, loc_feat=loc_feat,
                                                  reason='Location similar in: ', multi_flag=True)
            reason_db[sub_reason_col_name] = loc_comp_loc

            print('5. Is there a similar company inside the recommended location?')
            sub_reason_col_name = reason_col_name[4][0]
            if args.sampled or (cityname[ind_city] not in ['New York', 'San Francisco', 'Los Angeles']):
                recall_com5 = sub_rec_similar_company_v2(comp_loc=comp_loc, sspd=sspd, thresh=0.05, bid=bid, cid=cid)
                sim_comp_name = recall_com5.get_reason(comp_feat=comp_feat, comp_feat_col=comp_feat_col,
                                                       comp_feat_normed=comp_feat_normed,
                                                       reason_col_name=sub_reason_col_name)
            elif cityname[ind_city] in ['New York', 'San Francisco', 'Los Angeles']:
                recall_com5 = sub_rec_similar_company_v2(comp_loc=comp_loc, sspd=sspd, thresh=0.05)
                sim_comp_name = recall_com5.get_reason_batch(comp_feat=comp_feat, comp_feat_col=comp_feat_col,
                                                             comp_feat_normed=comp_feat_normed,
                                                             reason_col_name=sub_reason_col_name, batch_size=5000)
            print('==> Total pairs generated: %d' % len(sim_comp_name))
            reason_db[sub_reason_col_name] = sim_comp_name

            print('6. Close to current location')
            sub_reason_col_name = reason_col_name[5][0]
            recall_com6 = sub_rec_location_distance(reason_col_name=sub_reason_col_name)
            sub_close_loc = recall_com6.get_reason(sspd=sspd, loc_feat=loc_feat, comp_feat=comp_feat, dist_thresh=3.2e3)
            reason_db[sub_reason_col_name] = sub_close_loc

            print('7. Inventory bom')
            sub_reason_col_name = reason_col_name[6][0]
            invdb = pd.read_csv(pjoin(datapath, inventory_file))
            recall_com7 = sub_rec_inventory_bom(invdb = invdb, reason='Inventory reason: The available space of this location can hold your company.',bid=bid,cid=cid)
            sub_inventory_db = recall_com7.get_reason(sspd=sspd,comp_feat=comp_feat,comp_col='emp_here',inv_col='max_reservable_capacity',reason_col=sub_reason_col_name)
            reason_db[sub_reason_col_name] = sub_inventory_db
            print('==> Total pairs generated:%d'%len(sub_inventory_db))

            sample_sspd = sspd
            print('Merging reasons')
            for col_name,priority in reason_col_name:
                match_key = list(set([bid,cid]) & set(reason_db[col_name].columns)) #sometimes only location uuid is given
                sample_sspd = sample_sspd.merge(reason_db[col_name], on=match_key, how='left', suffixes=sfx)

            sample_sspd = sample_sspd.fillna('')
            print('Json format transforming...')
            sorted_reason_col_name = sorted(reason_col_name, key=lambda x: x[1])
            sorted_reason_col_name = [c[0] for c in sorted_reason_col_name]
            sample_sspd['reason'] = sample_sspd.apply(
                lambda x: merge_str_2_json_rowise_reformat(row=x, src_cols=sorted_reason_col_name, jsKey='reasons',
                                                           target_phss=['Location similar in: ', 'Implicit reason: ']),
                axis=1)

            sample_sspd[cid] = sample_sspd[cid].astype(int)
            sample_sspd = sample_sspd.rename(columns={
                "reason": "note", "duns_number": "company_id"
            })
            sample_sspd['building_id'] = sample_sspd['atlas_location_uuid'].apply(lambda x: hash(x))
            sample_sspd['algorithm'] = 'model_wide_and_deep'

            col_list = ['company_id', 'building_id', 'similarity', 'note', 'atlas_location_uuid', 'algorithm']
            sample_sspd = sample_sspd[col_list]
            sample_sspd['similarity'] = sample_sspd['similarity'].round(4)

            print(len(sample_sspd))

            sample_sspd.to_csv('sub_' + ssfile[ind_city])

##merging files

if not args.nomerge:
    print('merging results')
    if args.tt:
        dfs = pd.read_csv('sub_' + ssfile[testcode], index_col=0)
    else:
        dfs = []
        for filename in ssfile:
            dfs.append(pd.read_csv('sub_' + filename, index_col=0))
        dfs = pd.concat(dfs, axis=0).reset_index(drop=True)

    loc_df = dfs.groupby(bid, sort=True)[[bid]].first().reset_index(drop=True)

    k = list(range(len(loc_df)))
    pd_id = pd.DataFrame(np.array(k), columns=['building_id'])
    loc_df = pd.concat([loc_df, pd_id], axis=1)

    dfs = dfs[['company_id', 'similarity', 'note', 'algorithm', 'atlas_location_uuid']].merge(loc_df,
                                                                                              on='atlas_location_uuid',
                                                                                              how='left',
                                                                                              suffixes=['', '_right'])

    col_list = ['company_id', 'building_id', 'similarity', 'note', 'algorithm', 'atlas_location_uuid']
    dfs = dfs[col_list]

    if args.tt:
        dfs.to_csv('sub_all_similarity_multi_test.csv', index=False)
    else:
        dfs.to_csv('sub_all_similarity_multi.csv', index=False)

    print('Done!')
else:
    print('Done without merging, only mid files are generated!')
