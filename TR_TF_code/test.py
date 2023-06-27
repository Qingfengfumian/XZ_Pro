import time,os
import pandas as pd
import numpy as np
from datetime import datetime

import os
import shutil

new_path = '/home/XZ_Pro_AIOps/TR_TF_code/'

# for ftype in ['HW','ZX']:
#     for mode in ['4G','5G']:
#         try:os.mkdir(new_path+'/XZ_{}_{}'.format(mode,ftype))
#         except:print(new_path+'/XZ_{}_{}已经存在'.format(mode,ftype))
#         os.mkdir(new_path+'/XZ_{}_{}/LS_{}_{}'.format(mode,ftype,mode,ftype))
#         os.mkdir(new_path+'/XZ_{}_{}/LS_{}_{}/Data'.format(mode,ftype,mode,ftype))
#         os.mkdir(new_path+'/XZ_{}_{}/LS_{}_{}/materials'.format(mode,ftype,mode,ftype))

ftype = 'HW'
data_path = '/home/XZ_Pro_AIOps/Data/拉萨/Alert_Deal/Samp_华为/'
#data_path = '/home/AIOps/LN_Project/TR_TF_code/LN_4G_HW/QS_4G_HW/'
org_path = '/home/XZ_Pro_AIOps/TR_TF_code/XZ_4G_{}/LS_4G_{}/Data/'.format(ftype,ftype)
org_path1 = '/home/XZ_Pro_AIOps/TR_TF_code/XZ_5G_{}/LS_5G_{}/Data/'.format(ftype,ftype)
for ii in os.listdir(data_path):
    if '_delJZ_4G' in ii:
        shutil.copy(data_path+ii,org_path+ii)
    elif '_delJZ_5G' in ii:
        shutil.copy(data_path+ii,org_path1+ii)
