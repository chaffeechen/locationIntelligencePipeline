feature_column = {
    'not_feat_col': ['duns_number',
                     'atlas_location_uuid',
                     'longitude_loc',
                     'latitude_loc',
                     'city',
                     'label'],
    'cont_col_nameC': ['emp_here', 'emp_total', 'sales_volume_us', 'square_footage', 'emp_here_range'],
    'spec_col_nameC': 'emp_here_range',
    'cont_col_nameL': ['score_predicted_eo', 'score_employer', 'num_emp_weworkcore', 'num_poi_weworkcore',
                       'pct_wwcore_employee', 'pct_wwcore_business', 'num_retail_stores', 'num_doctor_offices',
                       'num_eating_places', 'num_drinking_places', 'num_hotels', 'num_fitness_gyms',
                       'population_density', 'pct_female_population', 'median_age', 'income_per_capita',
                       'pct_masters_degree', 'walk_score', 'bike_score'],
    'key_col_comp': ['duns_number'],
    'key_col_loc': ['atlas_location_uuid'],

    'dummy_col_nameL': ['building_class'],
    'dummy_col_nameC': ['major_industry_category', 'location_type', 'primary_sic_2_digit'],
}

cityabbr = ['PA', 'SF', 'SJ', 'LA', 'NY']
citylongname = ['Palo Alto', 'San Francisco', 'San Jose', 'Los Angeles', 'New York']

origin_comp_file = ['dnb_pa.csv', 'dnb_sf.csv', 'dnb_sj.csv', 'dnb_Los_Angeles.csv', 'dnb_New_York.csv']

inventory_file = 'inventory_bom.csv'
compstak_file = 'tetris_mv_tetris_transactions_2016_current.csv'
compstak_dnb_match_file = 'compstak_dnb_v2.csv'

salesforce_space = {
    'cid': 'account_id',
    'bid': 'atlas_location_uuid',

    'acc_db': 'salesforce_pair_account_info.csv',
    'visit_db': 'salesforce_pair.csv',
    'acc_city_db': 'salesforce_acc_city',
    'city_atlas_db': 'salesforce_city_atlas',

    'comp_feat_norm_param': 'salesforce_comp_feat_norm_param',
    'loc_feat_norm_param': 'salesforce_loc_feat_norm_param',
    'comp_feat_dummy_param': 'salesforce_comp_feat_dummy_param',
    'loc_feat_dummy_param': 'salesforce_loc_feat_dummy_param',

    'feature_column': {
        'not_feat_col': ['account_id',
                         'atlas_location_uuid',
                         'longitude_loc',
                         'latitude_loc',
                         'city',
                         'label'],
        'cont_col_nameC': ['emp_here', 'emp_total', 'sales_volume_us', 'square_footage', 'emp_here_range'],
        'cont_col_nameL': ['score_predicted_eo', 'score_employer', 'num_emp_weworkcore', 'num_poi_weworkcore',
                           'pct_wwcore_employee', 'pct_wwcore_business', 'num_retail_stores', 'num_doctor_offices',
                           'num_eating_places', 'num_drinking_places', 'num_hotels', 'num_fitness_gyms',
                           'population_density', 'pct_female_population', 'median_age', 'income_per_capita',
                           'pct_masters_degree', 'walk_score', 'bike_score'],
        'key_col_comp': ['duns_number'],
        'key_col_loc': ['atlas_location_uuid'],

        'dummy_col_nameL': ['building_class'],
        'dummy_col_nameC': ['major_industry_category', 'location_type', 'primary_sic_2_digit'],
    },
}
