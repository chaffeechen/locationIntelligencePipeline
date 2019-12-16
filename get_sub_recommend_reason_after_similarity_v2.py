import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import pandas as pd
import argparse
from utils import *
pjoin = os.path.join



if __name__ == '__main__':
    """
    Appending reason with score matrix
    """
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--run_root', default='/Users/yefeichen/Database/location_recommender_system/')
    arg('--ls_card',default='location_scorecard_191113.csv')
    arg('--apps',type=str,default='_191113.csv')
    arg('--sampled',action='store_true',help='is similarity score sampled?')
    arg('--ww',action='store_true',help='is ww building only?')

    args = parser.parse_args()

    datapath = args.run_root
    cfile = ['dnb_pa.csv','dnb_sf.csv','dnb_sj.csv','dnb_Los_Angeles.csv','dnb_New_York.csv']
    lfile = args.ls_card
    clfile = ['PA','SF','SJ','LA','NY']
    clfile = [c+args.apps for c in clfile]
    cityname = ['Palo Alto','San Francisco','San Jose','Los Angeles', 'New York']

    comp_feat_file = 'company_feat' + args.apps
    comp_feat_normed = pd.read_csv(pjoin(datapath, comp_feat_file), index_col=0)
    loc_feat_file = 'location_feat' + args.apps
    loc_feat_normed = pd.read_csv(pjoin(datapath, loc_feat_file), index_col=0)

    comp_feat_col = [c for c in comp_feat_normed.columns if c not in ['duns_number', 'atlas_location_uuid']]
    loc_feat_col = [c for c in loc_feat_normed.columns if c not in ['duns_number', 'atlas_location_uuid']]

    if args.ww:
        ww = 'ww_'
    else:
        ww = ''

    if args.sampled:
        sam = 'sampled_'
    else:
        sam = ''

    ssfile = [ sam+ww+c.replace(args.apps,'')+'_similarity'+args.apps for c in clfile ]
    # ssfile = ['ww_PA_similarity_20191113.csv','ww_SF_similarity_20191113.csv','ww_SJ_similarity_20191113.csv',
    #           'ww_LA_similarity_20191113.csv','ww_NY_similarity_20191113.csv']
    dlsub_ssfile = ['dlsub_'+c for c in ssfile]

    wework_location_only = args.ww

    for ind_city in range(5):
        print('##city: %s processing##'%cityname[ind_city])
        comp_feat = pd.read_csv(pjoin(datapath, cfile[ind_city]))
        comp_loc = pd.read_csv(pjoin(datapath, clfile[ind_city]))
        loc_feat = pd.read_csv(pjoin(datapath, lfile))

        # if we only focus on buildings with companies inside
        loc_feat = loc_feat.merge(comp_loc[['atlas_location_uuid']].groupby('atlas_location_uuid').first().reset_index(),
                                  on='atlas_location_uuid', how='inner', suffixes=['', '_right'])

        print('Generate reason files:<key,reason>:<locid,reason>,<(compid,locid),reason>')
        print('Filtering')
        # Global filter: master degree!
        global_ft = global_filter(loc_feat=loc_feat)
        sub_loc_feat = global_ft.city_filter(city_name=cityname[ind_city]).end()

        if wework_location_only:
            sub_loc_feat_ww = sub_loc_feat[sub_loc_feat['is_wework'] == True]
            sub_comp_loc = pd.merge(comp_loc, sub_loc_feat_ww[['atlas_location_uuid']], on='atlas_location_uuid',
                                    how='inner', suffixes=['', '_right'])
        else:
            sub_comp_loc = pd.merge(comp_loc, sub_loc_feat[['atlas_location_uuid']], on='atlas_location_uuid',
                                    how='inner', suffixes=['', '_right'])

        reason1 = 'reason_similar_biz' #sub_pairs
        reason2 = 'reason_location_based' #sub_loc_recall
        reason3 = 'reason_model_based' #dlsubdat
        reason4 = 'reason_similar_location'
        reason5 = 'reason_closest_company'
        reason6 = 'reason_close_2_current_location'
        print('Reasoning')
        # Reason 1:
        print('1. Customized Reason')
        # matching_col = 'major_industry_category'
        matching_col = 'primary_sic_2_digit_v2'
        recall_com1 = sub_rec_similar_company(comp_feat=comp_feat, comp_loc=sub_comp_loc,
                                              matching_col=matching_col, reason_col_name=reason1)
        sub_pairs = recall_com1.get_candidate_location_for_company(query_comp_feat=comp_feat, reason='like')
        #explanar
        sub_pairs[reason1] = sub_pairs.apply(lambda x: 'This location has a tenant company which is in the same industry as your company.', axis=1)
        print(sub_pairs.shape)

        # Reason2:
        print('2. Location based Reason')
        recall_com2 = sub_rec_condition(sub_loc_feat)
        sub_loc_recall_com2 = recall_com2.exfiltering('num_fitness_gyms', percentile=0.5, reason='there are enough gyms to work out',reason_col_name=reason2)
        sub_loc_recall_com3 = recall_com2.exfiltering('num_drinking_places', percentile=0.5,
                                                      reason='there are enough bars to have a drink', reason_col_name=reason2)
        sub_loc_recall_com4 = recall_com2.exfiltering('num_eating_places', percentile=0.5, reason='there are enough restaurants to get food',reason_col_name=reason2)
        print('recall_location_size: %d, %d, %d' % (
        len(sub_loc_recall_com2), len(sub_loc_recall_com3), len(sub_loc_recall_com4.shape)))

        print('Merging general reasons')
        sub_loc_recall = pd.concat([sub_loc_recall_com2, sub_loc_recall_com3, sub_loc_recall_com4], axis=0)
        # explanar:
        sub_loc_recall = merge_rec_reason_rowise(sub_loc_recall, group_cols=['atlas_location_uuid'],
                                                 merge_col=reason2,sep='| ')
        sub_loc_recall[reason2] = 'This building is at a location with great amenities: ' + sub_loc_recall[reason2] + '. '

        if wework_location_only:
            sub_loc_recall = sub_loc_recall.merge(sub_loc_feat_ww[['atlas_location_uuid']], on='atlas_location_uuid',
                                                  how='inner', suffixes=['', '_right'])
        print('sub_loc_recall sized %d' % len(sub_loc_recall))
        # sub_loc_recall.head()

        # Reason3:
        print('3. Model based Reason')
        featTranslator = feature_translate()
        dlsubdat = pd.read_csv(pjoin(datapath,dlsub_ssfile[ind_city]),index_col=0)
        dlsubdat[reason3] = dlsubdat.apply(lambda row:featTranslator.make_sense(row['merged_feat']),axis=1)
        dlsubdat = dlsubdat[['atlas_location_uuid','duns_number',reason3]]
        print('>> pairs: %d' % len(dlsubdat) )

        #### attach these with scores
        # topk = 300
        sspd = pd.read_csv(pjoin(datapath, ssfile[ind_city]), index_col=0)

        # sample_sspd = sspd.groupby('atlas_location_uuid').apply(lambda x: x.nlargest(topk, ['similarity'])).reset_index(
        #     drop=True)
        sample_sspd = sspd

        print('4. Similar location')
        cont_col_nameL = ['score_predicted_eo', 'score_employer', 'num_emp_weworkcore', 'num_poi_weworkcore',
                          'pct_wwcore_employee', 'pct_wwcore_business', 'num_retail_stores', 'num_doctor_offices',
                          'num_eating_places', 'num_drinking_places', 'num_hotels', 'num_fitness_gyms',
                          'population_density', 'pct_female_population', 'median_age', 'income_per_capita',
                          'pct_masters_degree', 'walk_score', 'bike_score']
        dummy_col_nameL = ['building_class']
        recall_com4 = sub_rec_similar_location(cont_col_name=cont_col_nameL, dummy_col_name=dummy_col_nameL,reason_col_name=reason4)
        loc_comp_loc = recall_com4.get_reason(sspd = sspd, comp_loc=comp_loc, loc_feat=loc_feat, reason='location similar in: ')

        print('5. Similar company name')
        recall_com5 = sub_rec_similar_company_v2(comp_loc = comp_loc, sspd = sspd , thresh=0.05)
        sim_comp_name = recall_com5.get_reason(comp_feat=comp_feat,comp_feat_col=comp_feat_col,comp_feat_normed=comp_feat_normed,reason_col_name = reason5 )

        print('6. Close to current location')
        recall_com6 = sub_rec_location_distance(reason_col_name=reason6)
        sub_close_loc = recall_com6.get_reason(sspd=sspd,loc_feat=loc_feat,comp_feat=comp_feat,dist_thresh=3.2e3)

        print(len(sample_sspd))
        #1 merge customized reason
        sample_sspd = pd.merge(sample_sspd, sub_pairs, on=['atlas_location_uuid', 'duns_number'], how='left',
                               suffixes=['', '_right'])
        #2 merge company similarity reason
        sample_sspd = pd.merge(sample_sspd, sim_comp_name, on=['atlas_location_uuid', 'duns_number'], how='left',
                               suffixes=['', '_right'])

        sample_sspd = sample_sspd[sample_sspd[reason1].notnull() | sample_sspd[reason5].notnull()]
        sample_sspd[[reason1, reason5]] = sample_sspd[[reason1, reason5]].fillna('')
        sample_sspd = merge_rec_reason_colwise(sample_sspd, cols=[reason1, reason5], dst_col='reason', sep='#')

        #3 merge location similarity reason
        sample_sspd = pd.merge(sample_sspd, loc_comp_loc, on=['atlas_location_uuid', 'duns_number'], how='left',
                               suffixes=['', '_right'])

        sample_sspd = sample_sspd[sample_sspd['reason'].notnull() | sample_sspd[reason4].notnull()]
        sample_sspd[['reason', reason4]] = sample_sspd[['reason', reason4]].fillna('')
        sample_sspd = merge_rec_reason_colwise(sample_sspd, cols=['reason', reason4], dst_col='reason', sep='#')

        #4 merge location base reason
        sample_sspd = pd.merge(sample_sspd, sub_loc_recall[['atlas_location_uuid', reason2]], on='atlas_location_uuid',
                               how='left', suffixes=['', '_right']).reset_index(drop=True)

        sample_sspd = sample_sspd[sample_sspd['reason'].notnull() | sample_sspd[reason2].notnull()]
        sample_sspd[['reason', reason2]] = sample_sspd[['reason', reason2]].fillna('')
        sample_sspd = merge_rec_reason_colwise(sample_sspd, cols=['reason', reason2], dst_col='reason',sep='#')

        #5 merge location distance
        sample_sspd = pd.merge(sample_sspd, sub_close_loc, on=['atlas_location_uuid', 'duns_number'], how='left',
                               suffixes=['', '_right'])

        sample_sspd = sample_sspd[sample_sspd['reason'].notnull() | sample_sspd[reason6].notnull()]
        sample_sspd[['reason', reason6]] = sample_sspd[['reason', reason6]].fillna('')
        sample_sspd = merge_rec_reason_colwise(sample_sspd, cols=['reason', reason6], dst_col='reason', sep='#')

        #6 merge model based reason
        sample_sspd = pd.merge(sample_sspd,dlsubdat, on=['atlas_location_uuid', 'duns_number'], how='left',
                               suffixes=['', '_right'])

        sample_sspd = sample_sspd[sample_sspd['reason'].notnull() | sample_sspd[reason3].notnull()]
        sample_sspd[['reason', reason3]] = sample_sspd[['reason', reason3]].fillna('')
        sample_sspd = merge_rec_reason_colwise(sample_sspd, cols=['reason', reason3], dst_col='reason',sep='#')





        print('json format transforming...')

        sample_sspd = reason_json_format(sample_sspd, col_name='reason',sep='#')

        sample_sspd['duns_number'] = sample_sspd['duns_number'].astype(int)
        sample_sspd = sample_sspd.rename(columns={
            "reason": "note", "duns_number": "company_id"
        })
        sample_sspd['building_id'] = sample_sspd['atlas_location_uuid'].apply(lambda x: hash(x))
        sample_sspd['algorithm'] = 'model_deep_and_wide'

        col_list = ['company_id', 'building_id', 'similarity', 'note', 'atlas_location_uuid', 'algorithm']
        sample_sspd = sample_sspd[col_list]

        print(len(sample_sspd))

        sample_sspd['similarity'] = sample_sspd['similarity'].round(4)

        sample_sspd.to_csv('sub_' + ssfile[ind_city])
        # sspd[sspd['reason_right'].isnull()]

##merging files
print('merging results')
dfs = []
for filename in ssfile:
    dfs.append(pd.read_csv('sub_'+filename,index_col=0))
dfs = pd.concat(dfs,axis=0).reset_index(drop=True)

loc_df = dfs.groupby('atlas_location_uuid',sort=True)[['atlas_location_uuid']].first().reset_index(drop=True)

k = list(range(len(loc_df)))
pd_id = pd.DataFrame(np.array(k),columns=['building_id'])
loc_df = pd.concat([loc_df,pd_id],axis=1)

dfs = dfs[['company_id', 'similarity', 'note', 'algorithm', 'atlas_location_uuid']].merge(loc_df,on='atlas_location_uuid',how='left',suffixes=['','_right'])

col_list = ['company_id', 'building_id', 'similarity', 'note', 'algorithm', 'atlas_location_uuid']
dfs = dfs[col_list]

dfs.to_csv('sub_all_similarity.csv',index=False)
