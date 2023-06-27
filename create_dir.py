import os
from utils import mkdir
import shutil


def run_create_dir(para):
    required_dir = ['./Data/汇总/Alert_Data/',
                    './Data/汇总/Project_Opt/',
                    './Data/汇总/Fault_Ord/',
                    './Data/汇总/Reserve_List/'
                    ]
    for distr in para.distr_list:
            # required_dir.append("./Data/{}/Project_Opt/".format(distr))
            required_dir.append('./Data/{}/Alert_Data/'.format(distr))
            required_dir.append('./Data/{}/Alert_Deal/'.format(distr))
            required_dir.append('./Data/{}/Inter_Result/'.format(distr))
            required_dir.append('./Data/{}/Alert_Samp/'.format(distr))
            required_dir.append('./Data/{}/Inspect_List/'.format(distr))
            # required_dir.append('./Data/{}/Reserve_List/'.format(distr))
            for ftype in para.ftype_list:
                for mod_de in ['4G','5G']:
                    required_dir.append('./Data/{}/Inter_Result/{}/{}_告警_标题/'.format(distr,mod_de, ftype))
                    required_dir.append('./Data/{}/Inter_Result/{}/{}_故障_标准化/'.format(distr,mod_de, ftype))
                    required_dir.append('./Data/{}/Inter_Result/{}/{}_特征标题/'.format(distr,mod_de, ftype))
                    required_dir.append('./Data/{}/Inter_Result/{}/{}_故障_模型/'.format(distr,mod_de, ftype))
                    required_dir.append('./Data/{}/Inter_Result/{}/{}_特征编码/'.format(distr, mod_de,ftype))
                    required_dir.append('./Data/{}/Inter_Result/{}/{}_退服_模型/'.format(distr,mod_de, ftype))
    for path in required_dir:
        if os.path.exists(path)==False:
            mkdir(path)
    # for distr in para.distr_list:
    #     if distr=='铜川':
    #         NEW_TC = './Data/铜川/Inter_Result/'
    #         OLD_TC = './Temp/铜川模型/'
    #         if os.path.exists(NEW_TC):
    #             shutil.rmtree(NEW_TC)
    #         shutil.copytree(OLD_TC, NEW_TC)
    #     else:
    #         NEW_HZ = './Data/{}/Inter_Result/'.format(distr)
    #         OLD_HZ = './Temp/汉中模型/'
    #         if os.path.exists(NEW_HZ):
    #             shutil.rmtree(NEW_HZ)
    #         shutil.copytree(OLD_HZ, NEW_HZ)
    #     new_opt = './Data/{}/Project_Opt/'.format(distr)
    #     old_opt = './Temp/Project_Opt/'.format(distr)
    #
    #     if os.path.exists(new_opt):
    #          shutil.rmtree(new_opt)
    #
    #     shutil.copytree(old_opt, new_opt)
