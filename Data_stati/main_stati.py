# -*- coding: utf-8 -*-
"""

@author: nzl
"""
# 导入库函数
import os
from warn_stats_convert import main
from data_deal import data_process_new

if __name__ == '__main__':
    if os.listdir('/data/resources/result_stati'):
        for f in os.listdir('/data/resources/result_stati'):
            path_file2 = os.path.join('/data/resources/result_stati', f)
            if os.path.isfile(path_file2):
                os.remove(path_file2)

    data_process_new()
    main()
