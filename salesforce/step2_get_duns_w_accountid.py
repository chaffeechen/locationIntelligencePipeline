import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import pandas as pd
import numpy as np

pjoin = os.path.join

from salesforce.header import *


sfdat = pd.read_csv(pjoin(datapath_sf,salesforce_file['opp_w_acc']))
dnbmtdat = pd.read_csv(pjoin(datapath_sf,salesforce_file['link_name_atlas_duns']),index_col=0)
dnbmtdat = dnbmtdat[['op_orig_name','atlas_location_uuid','duns_number']]
dnbcompdat = pd.read_csv(pjoin(datapath_sf,salesforce_file['dnb_duns_name']))

print('total records:%d'%len(sfdat))

sfdat = sfdat.merge(dnbmtdat,left_on=['atlas_location_uuid','Name'],
                    right_on=['atlas_location_uuid','op_orig_name'])

sfdat = sfdat.rename(columns={'op_orig_name':'sf_name','city':'loc_city'})
sfdat = sfdat[['Id','AccountId','atlas_location_uuid','loc_city','sf_name','duns_number']]

sfdat = sfdat.merge(dnbcompdat,on='duns_number').reset_index(drop=True)
sfdat = sfdat.rename(columns={'business_name':'dnb_name'})
print('total records matched with dnb:%d'%len(sfdat))

sfdat.to_csv(pjoin(datapath_sf,salesforce_file['opp_w_acc_duns']))

sfdatGrp = sfdat.groupby(['duns_number','city'])

duns_we_care = sfdatGrp.first().reset_index()[['duns_number','city']]

duns_we_care.to_csv(pjoin(datapath_sf,salesforce_file['duns_city_unique_in_opp']))


print('Get account and duns_number')
ac_dat = pd.read_csv(pjoin(datapath_sf,salesforce_file['acc']),error_bad_lines=False,header=None)
ac_dat = ac_dat[[8,9]]
ac_dat = ac_dat.rename(columns={8:'AccountId',9:'Name'})

ac_dat.head()
dnbmtdat2 = pd.read_csv(pjoin(datapath_sf,'op_loc_atlas_city_1210_good.csv'),index_col=0)
dnbmtdat2 = dnbmtdat2[['op_orig_name','duns_number','dnb_orig_name']].groupby(['op_orig_name','duns_number']).first().reset_index()
ac_duns_dat = ac_dat.merge(dnbmtdat2,left_on=['Name'],right_on='op_orig_name',suffixes=sfx)
ac_duns_dat = ac_duns_dat.rename(columns={'op_orig_name':'salesforce_name','dnb_orig_name':'dnb_name'})

ac_duns_dat.to_csv(pjoin(datapath_sf,salesforce_file['acc_duns']))