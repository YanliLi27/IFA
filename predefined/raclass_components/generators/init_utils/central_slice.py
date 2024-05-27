import SimpleITK as sitk
import numpy as np
from scipy.ndimage import binary_fill_holes
import os
from tqdm import tqdm


def _normalization(data: np.array) ->np.array:  
    # return the normalized data, the mean or max will not be reserved
    max_value = np.max(data)
    min_value = np.min(data)
    data = (data - min_value) / (max_value - min_value)
    return data  # here, data will be in the range of (0, 1)

def _central_n_slices(value_array: list, num:int=5)->int:
    # value_array: [value_of_mask-1, ...] [depth, 1]
    value_array = np.array(value_array)
    five_value_array = []
    for i in range(0, len(value_array)+1-num):
        value_around = np.sum(value_array[i:i+num])
        five_value_array.append(value_around)
    max_index = five_value_array.index(max(five_value_array))
    sorted_index = sorted(range(len(five_value_array)), key=lambda k:five_value_array[k], reverse=True)
    # return a list with length = 20 - num, [a, b, c, d, e, ...] a represents the index of the largest value in array
    # the first one is the largest one
    order_array = [sorted_index.index(i) for i in range(len(sorted_index))]
    # become the order for each slice, in the order of 0-(20-num)
    return max_index, order_array


def _square_selector(data:np.array)->str:  # data: [20, 512, 512]
    assert len(data.shape) == 4 or len(data.shape) == 3 # "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[:], dtype=bool)  # [512, 512] 因为20层应该是要全用，所以说完全可以把z轴作为channel
    value_in_mask = []   # 20 slices with 20 sums
    for c in range(data.shape[0]):  # 读取单个slice
        nonzero_mask[c] = data[c] >= 0.05   # threshold = 0.1,  out of range(0, 1) after normalization
        nonzero_mask[c] = binary_fill_holes(nonzero_mask[c])  # 输入数据已经是处理后的数据了，此时无需再使用图形学
        value_in_mask.append(np.sum(nonzero_mask[c]))
    max_range, oa = _central_n_slices(value_in_mask, num=5)
    max_range2, oa2 = _central_n_slices(value_in_mask, num=7)
    return max_range, max_range2, oa, oa2


def _hist_selector(data:np.array)->str:  # data: [20, 512, 512]
    assert len(data.shape) == 4 or len(data.shape) == 3 # "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    # nonzero_mask = np.zeros(data.shape[:], dtype=bool)  # [512, 512] 因为20层应该是要全用，所以说完全可以把z轴作为channel
    value_in_mask = []   # 20 slices with 20 sums
    for c in range(data.shape[0]):  # 读取单个slice
        hist, _ = np.histogram(data[c], bins=20, range=(0,1))
        value_in_mask.append(-np.std(hist))  # 取std作为均衡化的标准，越小越好因此取负值
    max_range, oa = _central_n_slices(value_in_mask, num=5)
    max_range2, oa2 = _central_n_slices(value_in_mask, num=7)
    return max_range, max_range2, oa, oa2


def central_selector(datapath:str)->str:
    # datapath: os.path.join(dirpath, file) -- data_root + subnaem + filename = 'Root/EAC_Wrist_TRA/Names'
    data = sitk.ReadImage(datapath)
    data_array = sitk.GetArrayFromImage(data)  # output [slice, 512, 512]
    data_array = _normalization(data_array)  # 可以使用class内的normalization
    square_mr, square_mr2, square_oa, square_oa2 = _square_selector(data_array)  # [slice, 512, 512] the mask for each slice
    hist_mr, hist_mr2, hist_oa, hist_oa2 = _hist_selector(data_array)
    if square_mr==hist_mr:
        max_range = square_mr
    else:
        oa = list(np.asarray(square_oa) + np.asarray(hist_oa))
        max_range = oa.index(min(oa))
    
    if square_mr2==hist_mr2:
        max_range2 = square_mr2
    else:
        oa2 = list(np.asarray(square_oa2) + np.asarray(hist_oa2))
        max_range2 = oa2.index(min(oa2))

    userange = f':{max_range}to{max_range+5}plus{max_range2}to{max_range2+7}'
    return userange  # 返回一个范围的上下限，然后用':10to15plus5to15'来保存
    

def central_slice_generator(data_root:str, common_list:dict, target_category:list)->dict:
    # common_list is the dict of EAC/CSA/ATL - of all Wrist/MCP/Foot, that patients exist in all target_site,
    # {'EAC':[LIST], 'CSA':[LIST], 'ATL':[LIST]}
    if not target_category:
        default_target_category = ['EAC', 'CSA', 'ATL']
    else:
        default_target_category = target_category
    default_target_site = ['Wrist', 'MCP', 'Foot']
    default_target_dirc = ['TRA', 'COR']
    cs_dict = {}  # save the {'EAC_Wrist_TRA':[LIST-'Names_label.mha:CentralSlice'], ..., 'CSA_MCP_COR':[LIST-'Names_label.mha:CentralSlice'], ...}
    # Step. 1 the list of subdirectory, to find the file
    subdir = []
    for cate in default_target_category:
        for site in default_target_site:
            for dirc in default_target_dirc:
                subdir.append(f'{cate}_{site}_{dirc}')
    # subdir ['EAC_Wrist_TRA', ...]
    print(subdir)

    # Step. 2 在子目录列表的对象中循环，计算结果，并添加到名称末尾，由此名称变为：子目录名/文件名_label.mha:central slice。
    for subname in subdir:  # subname 'EAC_Wrist_TRA'
        sub_extended_list = []
        dirpath = os.path.join(data_root, subname)
        filelist = os.listdir(dirpath)  # ['file_names']
        for file in tqdm(filelist):
            id = file.split('-')[2]
            cate = id.split('_')[1]
            if id in common_list[cate]:
                central_slice_range = central_selector(os.path.join(dirpath, file))  # str ':10to15'
                name_str = file + central_slice_range  # 'Names_1.mha:10to15plus5to15'
                sub_extended_list.append(name_str)  # 'Names_1.mha:10to15plus5to15'
        cs_dict[subname] = sub_extended_list
    return cs_dict




        