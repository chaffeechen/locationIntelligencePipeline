import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import pandas as pd
import numpy as np
from salesforce.header import *

pjoin = os.path.join

op_dat = pd.read_csv(pjoin(datapath_sf, salesforce_file['opp']),header=None,names=['Id','AccountId','atlas_location_uuid','n1','date','n2','n3','n4'])
op_dat = op_dat[['Id','AccountId','atlas_location_uuid']]

loc_dat = pd.read_csv(pjoin(datapath,'location_scorecard_191113.csv'),index_col=0)
loc_dat = loc_dat[['atlas_location_uuid','city']]
print('Len of original opp table:%d'%len(op_dat))

op_dat = op_dat.merge(loc_dat,on='atlas_location_uuid',how='inner',suffixes=sfx)
print('Len of opp table cleaned with locations in location scorecard:%d'%len(op_dat))

ac_dat = pd.read_csv(pjoin(datapath_sf,salesforce_file['acc']),error_bad_lines=False,header=None)
ac_dat = ac_dat[[8,9]]
ac_dat = ac_dat.rename(columns={8:'AccountId',9:'Name'})

op_dat = op_dat.merge(ac_dat,on='AccountId',how='inner',suffixes=sfx)
print('Len of opp table attached with account:%d'%len(op_dat))

op_dat.to_csv(pjoin(datapath_sf,salesforce_file['opp_w_acc']),index=False)
"""
Id AccountId atlas_location_uuid city(location) Name
"""