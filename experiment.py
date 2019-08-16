# zr 0 ######################################################
import os

from utils import radar_data_grapher_volumned, generate_path

radarData_path = 'F:/indexPen/data/f_data_zr_0/f_data.p'
videoData_path = 'F:/indexPen/data/v_data_zr_0/cam2'
mergedImg_path = 'F:/indexPen/figures/zr_0'
out_path = 'F:/indexPen/csv/zr_0'
# zr 1 ######################################################
# radarData_path = 'F:/indexPen/data/f_data_zr_1/f_data.p'
# videoData_path = 'F:/indexPen/data/v_data_zr_1/cam2'
# mergedImg_path = 'F:/indexPen/figures/zr_1'
# out_path = 'F:/indexPen/csv/zr_1'

# py 0 ######################################################
# radarData_path = 'F:/indexPen/data/f_data_py_0/f_data.p'
# videoData_path = 'F:/indexPen/data/v_data_py_0/cam2'
# mergedImg_path = 'F:/indexPen/figures/py_0'
# out_path = 'F:/indexPen/csv/py_0'
# py 1 ######################################################
# radarData_path = 'F:/indexPen/data/f_data_py_1/f_data.p'
# videoData_path = 'F:/indexPen/data/v_data_py_1/cam2'
# mergedImg_path = 'F:/indexPen/figures/py_1'
# out_path = 'F:/indexPen/csv/py_1'

# ya 0 ######################################################
# radarData_path = 'F:/indexPen/data/f_data_ya_0/f_data.p'
# videoData_path = 'F:/indexPen/data/v_data_ya_0/cam2'
# mergedImg_path = 'F:/indexPen/figures/ya_0'
# out_path = 'F:/indexPen/csv/ya_0'
# ya 1 ######################################################
# radarData_path = 'F:/indexPen/data/f_data_ya_1/f_data.p'
# videoData_path = 'F:/indexPen/data/v_data_ya_1/cam2'
# mergedImg_path = 'F:/indexPen/figures/ya_1'
# out_path = 'F:/indexPen/csv/ya_1'
# ya 2 ######################################################
# radarData_path = 'F:/indexPen/data/f_data_ya_2/f_data.p'
# videoData_path = 'F:/indexPen/data/v_data_ya_2/cam2'
# mergedImg_path = 'F:/indexPen/figures/ya_2'
# out_path = 'F:/indexPen/csv/ya_2'
# ya 3 ######################################################
# radarData_path = 'F:/indexPen/data/f_data_ya_3/f_data.p'
# videoData_path = 'F:/indexPen/data/v_data_ya_3/cam2'
# mergedImg_path = 'F:/indexPen/figures/ya_3'
# out_path = 'F:/indexPen/csv/ya_3'

# zl 0 ######################################################
# radarData_path = 'F:/indexPen/data/f_data_zl_0/f_data.p'
# videoData_path = 'F:/indexPen/data/v_data_zl_0/cam2'
# mergedImg_path = 'F:/indexPen/figures/zl_0'
# out_path = 'F:/indexPen/csv/zl_0'
# zl 1 ######################################################
# radarData_path = 'F:/indexPen/data/f_data_zl_1/f_data.p'
# videoData_path = 'F:/indexPen/data/v_data_zl_1/cam2'
# mergedImg_path = 'F:/indexPen/figures/zl_1'
# out_path = 'F:/indexPen/csv/zl_1'
# # zl 2 ######################################################
# radarData_path = 'F:/indexPen/data/f_data_zl_2/f_data.p'
# videoData_path = 'F:/indexPen/data/v_data_zl_2/cam2'
# mergedImg_path = 'F:/indexPen/figures/zl_2'
# out_path = 'F:/indexPen/csv/zl_2'
# # zl 3 ######################################################
# radarData_path = 'F:/indexPen/data/f_data_zl_3/f_data.p'
# videoData_path = 'F:/indexPen/data/v_data_zl_3/cam2'
# mergedImg_path = 'F:/indexPen/figures/zl_3'
# out_path = 'F:/indexPen/csv/zl_3'

# zy 0 ######################################################
# radarData_path = 'F:/indexPen/data/f_data_zy_0/f_data.p'
# videoData_path = 'F:/indexPen/data/v_data_zy_0/cam2'
# mergedImg_path = 'F:/indexPen/figures/zy_0'
# out_path = 'F:/indexPen/csv/zy_0'
# zy 1 ######################################################
# radarData_path = 'F:/indexPen/data/f_data_zy_1/f_data.p'
# videoData_path = 'F:/indexPen/data/v_data_zy_1/cam2'
# mergedImg_path = 'F:/indexPen/figures/zy_1'
# out_path = 'F:/indexPen/csv/zy_1'
# zy 2 ######################################################
# radarData_path = 'F:/indexPen/data/f_data_zy_2/f_data.p'
# videoData_path = 'F:/indexPen/data/v_data_zy_2/cam2'
# mergedImg_path = 'F:/indexPen/figures/zy_2'
# out_path = 'F:/indexPen/csv/zy_2'
# zy 3 ######################################################
# radarData_path = 'F:/indexPen/data/f_data_zy_3/f_data.p'
# videoData_path = 'F:/indexPen/data/v_data_zy_3/cam2'
# mergedImg_path = 'F:/indexPen/figures/zy_3'
# out_path = 'F:/indexPen/csv/zy_3'



specimen_list = {generate_path('zr', 0), generate_path('zr', 1), generate_path('py', 0), generate_path('py', 1),
                 generate_path('ya', 0), generate_path('ya', 1), generate_path('zl', 0), generate_path('zl', 1),
                 generate_path('zy', 0), generate_path('zy', 1)}

# use data augmentation
# method for augmenting point cloud:
#
for path in specimen_list:
    radar_data_grapher_volumned(path)