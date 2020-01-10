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
    arg('--otversion', type=str, default='_200106.csv', help='output file\'s version')
    arg('--single',type=int,default=-1)
    arg('--merge', action='store_true', help='doing merge only')
    arg('--nomerge',action='store_true',help='dont merge')
    arg('--sampled', action='store_true', help='is similarity score sampled?')
    arg('--ww', action='store_true', help='is ww building only?')
    arg('--tt', action='store_true', help='doing test for 1 city only')
    arg('--ttcode',type=int,default=2)


    args = parser.parse_args()
    testcode = args.ttcode  # in --tt mode, which city is used for test
    if args.single >= 0:
        singlecode = min(len(citylongname),args.single)
        print('#####Attention#####')
        print('Only %s will be generated'%citylongname[singlecode])

    datapath = args.run_root
    if args.tt:
        datapath_mid = pjoin(datapath, 'tmp_table')
    else:
        datapath_mid = pjoin(datapath,'reason_table')
    cfile = origin_comp_file
    lfile = args.ls_card
    clfile = [c + args.apps for c in cityabbr]
    cityname = citylongname

    comp_feat_file = 'company_feat' + args.apps
    comp_feat_normed = pd.read_csv(pjoin(datapath, comp_feat_file), index_col=0)
    loc_feat_file = 'location_feat' + args.apps
    loc_feat_normed = pd.read_csv(pjoin(datapath, loc_feat_file), index_col=0)

    compstak_db = pd.read_csv(pjoin(datapath,compstak_file))[['tenant_id', 'expiration_date','city']]
    compstak_dnb = pd.read_csv(pjoin(datapath,compstak_dnb_match_file))[['tenant_id',cid,'city']]

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
    rsfile = ['z_reason_' + c + '_similarity' + args.otversion for c in cityabbr]
    dlsub_ssfile = ['dlsub_' + c for c in ssfile]

    wework_location_only = args.ww

    """
    Reasons cannot be duplicated because they are used as column names.
    (reason_col_name, priority) 
    priority 1 is highest.
    """
    reason_col_name = [
        ('reason_similar_biz', 1,True),  # sub_pairs
        ('reason_location_based', 7,True),  # sub_loc_recall
        ('reason_model_based', 8,True),  # dlsubdat
        ('reason_similar_location', 6,True),
        ('reason_similar_company', 5,True),
        ('reason_close_2_current_location', 2,True),
        ('reason_inventory_bom',3,True),
        ('reason_compstak',4,True),
    ]

    if args.tt:
        print('#####Attention#####')
        print('This is TEST mode only. Not PROD-DEV.')
    else:
        print('This is PROD-DEV mode.')

    if args.merge:
        print('Skip to merging directly...')
    else:
        for ind_city in range(len(cityname)):
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
                print('==> %d locations inside the city'%len(sub_loc_feat_ww))
            else:
                sub_comp_loc = pd.merge(comp_loc, sub_loc_feat[[bid]], on=bid,
                                        suffixes=sfx)  # multi comp loc

            sspd = pd.read_csv(pjoin(datapath, ssfile[ind_city]), index_col=0)
            total_pairs_num = len(sspd)
            print('Loading %d similarity score pairs' % total_pairs_num )

            print('Begin generating reasons')
            # Reason 1:
            print('1. Is there a company with similar biz inside the location?')
            sub_reason_col_name,_,usedFLG = reason_col_name[0]
            sub_reason_file_name = cityabbr[ind_city] + '_' + sub_reason_col_name + args.otversion
            sub_reason_file = pjoin(datapath_mid,sub_reason_file_name)

            if usedFLG:
                matching_col = 'primary_sic_4_digit' #'primary_sic_2_digit_v2','major_industry_category'
                query_comp_loc = sspd[[bid, cid]]
                query_comp_loc = query_comp_loc.merge(comp_feat[[cid, matching_col]], on=cid, suffixes=sfx)

                recall_com1 = sub_rec_similar_company(comp_feat=comp_feat, comp_loc=sub_comp_loc,
                                                      matching_col=matching_col, reason_col_name=sub_reason_col_name,
                                                      bid=bid, cid=cid,cname='business_name')

                sub_pairs = recall_com1.get_candidate_location_for_company_fast(query_comp_loc=query_comp_loc, reason='This location has a tenant company(%s) which is in the same industry as your company.')
                # explanar
                reason_db[sub_reason_col_name] = sub_pairs
                print('==> Coverage: %1.2f' % (len(reason_db[sub_reason_col_name])/total_pairs_num) )
                reason_db[sub_reason_col_name].to_csv(sub_reason_file)
            else:
                if os.path.isfile( sub_reason_file ):
                    reason_db[sub_reason_col_name] = pd.read_csv(sub_reason_file,index_col=0)
                    print('==> Load existing result with coverage: %1.2f' % (len(reason_db[sub_reason_col_name])/total_pairs_num) )
                else:
                    print('==> Skip')

            # Reason2:
            print('2. How is region?(Location based reason)')
            sub_reason_col_name, _, usedFLG = reason_col_name[1]
            sub_reason_file_name = cityabbr[ind_city] + '_' + sub_reason_col_name + args.otversion
            sub_reason_file = pjoin(datapath_mid,sub_reason_file_name)

            if usedFLG:
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
                print('==> %d, %d, %d will be merged' % (
                    len(sub_loc_recall_com2), len(sub_loc_recall_com3), len(sub_loc_recall_com4)))

                sub_loc_recall = pd.concat([sub_loc_recall_com2, sub_loc_recall_com3, sub_loc_recall_com4], axis=0)
                # print('==>%d'%len(sub_loc_recall))
                if wework_location_only:
                    sub_loc_recall = sub_loc_recall.merge(sub_loc_feat_ww[[bid]], on=bid,
                                                          suffixes=sfx)
                    # print('==>%d' % len(sub_loc_recall))
                # explanar:merge_rec_reason_rowise 需要在结尾加"."
                sub_loc_recall = merge_rec_reason_rowise(sub_loc_recall, group_cols=[bid],
                                                         merge_col=sub_reason_col_name, sep='. ')
                sub_loc_recall[sub_reason_col_name] = 'This building is at a location with great amenities: ' + sub_loc_recall[
                    sub_reason_col_name] + '. '

                reason_db[sub_reason_col_name] = sub_loc_recall
                print('==> Coverage: %1.2f' % (len(reason_db[sub_reason_col_name])/len(sub_loc_feat_ww) ))
                reason_db[sub_reason_col_name].to_csv(sub_reason_file)
            else:
                if os.path.isfile( sub_reason_file ):
                    reason_db[sub_reason_col_name] = pd.read_csv(sub_reason_file,index_col=0)
                    print('==> Load existing result with coverage: %1.2f' % (len(reason_db[sub_reason_col_name]) / total_pairs_num))
                else:
                    print('==> Skip')

            # Reason3: Tag!!!!
            print('3. Model based Reason(Implicit reason)')
            sub_reason_col_name, _, usedFLG = reason_col_name[2]
            sub_reason_file_name = cityabbr[ind_city] + '_' + sub_reason_col_name + args.otversion
            sub_reason_file = pjoin(datapath_mid,sub_reason_file_name)

            if usedFLG:
                featTranslator = feature_translate()
                dlsubdat = pd.read_csv(pjoin(datapath, dlsub_ssfile[ind_city]), index_col=0)
                dlsubdat[sub_reason_col_name] = dlsubdat.apply(lambda row: featTranslator.make_sense(row['merged_feat']),
                                                               axis=1)
                dlsubdat = dlsubdat[[bid, cid, sub_reason_col_name]]

                reason_db[sub_reason_col_name] = dlsubdat
                print('==> Coverage: %1.2f' % (len(reason_db[sub_reason_col_name]) / total_pairs_num))
                reason_db[sub_reason_col_name].to_csv(sub_reason_file)
            else:
                if os.path.isfile(sub_reason_file):
                    reason_db[sub_reason_col_name] = pd.read_csv(sub_reason_file, index_col=0)
                    print('==> Load existing result with coverage: %1.2f' % (len(reason_db[sub_reason_col_name]) / total_pairs_num))
                else:
                    print('==> Skip')


            print('4. Is the recommended location similar with its current one?')
            sub_reason_col_name, _, usedFLG = reason_col_name[3]
            sub_reason_file_name = cityabbr[ind_city] + '_' + sub_reason_col_name + args.otversion
            sub_reason_file = pjoin(datapath_mid,sub_reason_file_name)

            if usedFLG:
                cont_col_nameL = feature_column['cont_col_nameL']
                dummy_col_nameL = feature_column['dummy_col_nameL']
                recall_com4 = sub_rec_similar_location(cont_col_name=cont_col_nameL, dummy_col_name=dummy_col_nameL,
                                                       reason_col_name=sub_reason_col_name, cid=cid, bid=bid)
                loc_comp_loc = recall_com4.get_reason(sspd=sspd, comp_loc=comp_loc, loc_feat=loc_feat,
                                                      reason='Location similar in: ', multi_flag=True)
                reason_db[sub_reason_col_name] = loc_comp_loc
                print('==> Coverage: %1.2f' % (len(reason_db[sub_reason_col_name]) / total_pairs_num))
                reason_db[sub_reason_col_name].to_csv(sub_reason_file)
            else:
                if os.path.isfile(sub_reason_file):
                    reason_db[sub_reason_col_name] = pd.read_csv(sub_reason_file, index_col=0)
                    print('==> Load existing result with coverage: %1.2f' % (len(reason_db[sub_reason_col_name]) / total_pairs_num))
                else:
                    print('==> Skip')

            print('5. Is there a similar company inside the recommended location?')
            sub_reason_col_name, _, usedFLG = reason_col_name[4]
            sub_reason_file_name = cityabbr[ind_city] + '_' + sub_reason_col_name + args.otversion
            sub_reason_file = pjoin(datapath_mid,sub_reason_file_name)

            if usedFLG:
                matching_col = 'primary_sic_6_digit'
                query_comp_loc = sspd[[bid, cid]]
                query_comp_loc = query_comp_loc.merge(comp_feat[[cid, matching_col]], on=cid, suffixes=sfx)

                recall_com5_ext = sub_rec_similar_company(comp_feat=comp_feat, comp_loc=sub_comp_loc,
                                                      matching_col=matching_col, reason_col_name=sub_reason_col_name,
                                                      bid=bid, cid=cid,cname='business_name')
                sub_sspd = recall_com5_ext.get_candidate_location_for_company_fast(query_comp_loc=query_comp_loc, reason='This location has a tenant company(%s) which is in the same industry as your company.')
                # explanar
                sub_sspd = sspd.merge(sub_sspd[[cid,bid]],on=[cid,bid],suffixes=sfx)
                print('Shrinkage ratio: %1.2f' % (len(sub_sspd)/len(sspd)) )
                recall_com5 = sub_rec_similar_company_v2(comp_loc=comp_loc, sspd=sub_sspd, thresh=0.05)
                sim_comp_name = recall_com5.get_reason_batch(comp_feat=comp_feat, comp_feat_col=comp_feat_col,
                                                             comp_feat_normed=comp_feat_normed,
                                                             reason_col_name=sub_reason_col_name, batch_size=5000)
                reason_db[sub_reason_col_name] = sim_comp_name
                print('==> Coverage: %1.2f' % (len(reason_db[sub_reason_col_name]) / total_pairs_num))
                reason_db[sub_reason_col_name].to_csv(sub_reason_file)
                del sub_sspd
            else:
                if os.path.isfile(sub_reason_file):
                    reason_db[sub_reason_col_name] = pd.read_csv(sub_reason_file, index_col=0)
                    print('==> Load existing result with coverage: %1.2f' % (
                    len(reason_db[sub_reason_col_name]) / total_pairs_num))
                else:
                    print('==> Skip')

            print('6. Close to current location')
            sub_reason_col_name, _, usedFLG = reason_col_name[5]
            sub_reason_file_name = cityabbr[ind_city] + '_' + sub_reason_col_name + args.otversion
            sub_reason_file = pjoin(datapath_mid,sub_reason_file_name)

            if usedFLG:
                recall_com6 = sub_rec_location_distance(reason_col_name=sub_reason_col_name)
                sub_close_loc = recall_com6.get_reason(sspd=sspd, loc_feat=loc_feat, comp_feat=comp_feat, dist_thresh=3.2e3)
                reason_db[sub_reason_col_name] = sub_close_loc
                print('==> Coverage: %1.2f' % (len(reason_db[sub_reason_col_name]) / total_pairs_num))
                reason_db[sub_reason_col_name].to_csv(sub_reason_file)
            else:
                if os.path.isfile(sub_reason_file):
                    reason_db[sub_reason_col_name] = pd.read_csv(sub_reason_file, index_col=0)
                    print('==> Load existing result with coverage: %1.2f' % (len(reason_db[sub_reason_col_name]) / total_pairs_num))
                else:
                    print('==> Skip')

            print('7. Inventory bom')
            sub_reason_col_name, _, usedFLG = reason_col_name[6]
            sub_reason_file_name = cityabbr[ind_city] + '_' + sub_reason_col_name + args.otversion
            sub_reason_file = pjoin(datapath_mid,sub_reason_file_name)

            if usedFLG:
                invdb = pd.read_csv(pjoin(datapath, inventory_file))
                recall_com7 = sub_rec_inventory_bom(invdb = invdb, reason='Inventory reason: The available space of this location can hold your company.',bid=bid,cid=cid)
                sub_inventory_db = recall_com7.get_reason(sspd=sspd,comp_feat=comp_feat,comp_col='emp_here',inv_col='max_reservable_capacity',reason_col=sub_reason_col_name)
                reason_db[sub_reason_col_name] = sub_inventory_db
                print('==> Coverage: %1.2f' % (len(reason_db[sub_reason_col_name]) / total_pairs_num))
                reason_db[sub_reason_col_name].to_csv(sub_reason_file)
            else:
                if os.path.isfile(sub_reason_file):
                    reason_db[sub_reason_col_name] = pd.read_csv(sub_reason_file, index_col=0)
                    print('==> Load existing result with coverage: %1.2f' % (len(reason_db[sub_reason_col_name]) / total_pairs_num))
                else:
                    print('==> Skip')

            print('8. Compstak')
            sub_reason_col_name,_,usedFLG = reason_col_name[7]
            sub_reason_file_name = cityabbr[ind_city] + '_' + sub_reason_col_name + args.otversion
            sub_reason_file = pjoin(datapath_mid,sub_reason_file_name)

            if usedFLG:
                compstak_db_city = compstak_db.loc[compstak_db['city']==cityname[ind_city],:]
                compstak_dnb_city = compstak_dnb.loc[compstak_dnb['city'] == cityname[ind_city], :]
                recall_com8 = sub_rec_compstak(cpstkdb=compstak_db_city,cpstkdnb=compstak_dnb_city,
                                               reason = 'Compstak reason: The lease will expire in %d months.',
                                               cid=cid,bid=bid)
                sub_compstak_db = recall_com8.get_reason(sspd=sspd,reason_col=sub_reason_col_name)
                reason_db[sub_reason_col_name] = sub_compstak_db
                print('==> Coverage: %1.2f' % (len(reason_db[sub_reason_col_name]) / total_pairs_num))
                reason_db[sub_reason_col_name].to_csv(sub_reason_file)
            else:
                if os.path.isfile(sub_reason_file):
                    reason_db[sub_reason_col_name] = pd.read_csv(sub_reason_file, index_col=0)
                    print('==> Load existing result with coverage: %1.2f' % (len(reason_db[sub_reason_col_name]) / total_pairs_num))
                else:
                    print('==> Skip')

            """
            Merge reason for each city
            """
            sample_sspd = sspd
            print('Merging reasons')
            for col_name,priority,usedFLG in reason_col_name:
                if usedFLG:
                    match_key = list(set([bid,cid]) & set(reason_db[col_name].columns)) #sometimes only location uuid is given
                    sample_sspd = sample_sspd.merge(reason_db[col_name], on=match_key, how='left', suffixes=sfx)

            sample_sspd = sample_sspd.fillna('')
            print('Json format transforming...')
            sorted_reason_col_name = sorted(reason_col_name, key=lambda x: x[1])
            sorted_reason_col_name = [c[0] for c in sorted_reason_col_name]
            sample_sspd['reason'] = sample_sspd.apply(
                lambda x: merge_str_2_json_rowise_reformat(row=x, src_cols=sorted_reason_col_name, jsKey='reasons',
                                                           target_phss=['Location similar in: ', 'Implicit reason: ']), axis=1)

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

            sample_sspd.to_csv(pjoin(datapath_mid,rsfile[ind_city]))

##merging files

if not args.nomerge:
    print('merging results')
    if args.tt:
        dfs = pd.read_csv(pjoin(datapath_mid,rsfile[testcode]), index_col=0)
    else:
        dfs = []
        for filename in rsfile:
            dfs.append(pd.read_csv(pjoin(datapath_mid,filename), index_col=0))
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
        dfs.to_csv('sub_all_similarity_multi_test'+args.otversion, index=False)
    else:
        dfs.to_csv('sub_all_similarity_multi'+args.otversion, index=False)

    print('Done!')
else:
    print('Done without merging, only mid files are generated!')
