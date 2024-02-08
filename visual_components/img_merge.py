import sys
import cv2
import numpy as np
import os

root_path = r'D:\ESMIRA\CAM_auc\xgradcam'

data_path = os.path.join(root_path, 'cam_csa_result')
data_atl_path = os.path.join(root_path, 'cam_atl_result')
data_list = os.listdir(data_path)
data_ori_list = []
data_csa_list = []
for item in data_list:
    if 'cam' not in item:
        data_ori_list.append(item)
for item in data_list:
    if 'cam' in item:
        data_csa_list.append(item)

data_atl_list_0 = os.listdir(data_atl_path)
data_atl_list = []
for item in data_atl_list_0:
    if 'cam' in item:
        data_atl_list.append(item)

output_dir = os.path.join(root_path, 'combined')


for item in range(len(data_ori_list)):
    orginal_img_path = data_ori_list[item]
    cam_img_path = data_csa_list[item]
    cam_img_path_atl = data_atl_list[item]
    if 'cam' in orginal_img_path:
        print(orginal_img_path)
        sys.exit()
    elif 'cam' not in cam_img_path:
        print(cam_img_path)
        sys.exit()
    elif 'cam' not in cam_img_path_atl:
        print('atl')
        print(cam_img_path_atl)
        sys.exit()

    full_ori_path = os.path.join(data_path, orginal_img_path)
    full_cam_path = os.path.join(data_path, cam_img_path)
    full_cam_atl_path = os.path.join(data_atl_path, cam_img_path_atl)

    ori_img = cv2.imread(full_ori_path)
    cam_img = cv2.imread(full_cam_path)
    cam_atl_img = cv2.imread(full_cam_atl_path)
    combined_img = cv2.hconcat([ori_img, cam_img, cam_atl_img])

    output_name = '{}'.format(orginal_img_path)
    output_path = os.path.join(output_dir, output_name)
    cv2.imwrite(output_path, combined_img)
#
