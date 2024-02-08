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

def _central3_slices(value_array: list)->int:
    # value_array: [value_of_mask-1, ...] [depth, 1]
    value_array = np.array(value_array)
    five_value_array = []
    for i in range(0, len(value_array)-4):
        value_around = np.sum(value_array[i:i+5])
        five_value_array.append(value_around)
    max_index = five_value_array.index(max(five_value_array))
    max_range = f':{max_index}to{max_index+5}'
    return max_range

def _create_nonzero_mask(data:np.array)->str:  # data: [20, 512, 512]
    assert len(data.shape) == 4 or len(data.shape) == 3 # "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[:], dtype=bool)  # [512, 512] 因为20层应该是要全用，所以说完全可以把z轴作为channel
    value_in_mask = []   # 20 slices with 20 sums
    for c in range(data.shape[0]):  # 读取单个slice
        nonzero_mask[c] = data[c] >= 0.05   # threshold = 0.1,  out of range(0, 1) after normalization
        nonzero_mask[c] = binary_fill_holes(nonzero_mask[c])  # 输入数据已经是处理后的数据了，此时无需再使用图形学
        value_in_mask.append(np.sum(nonzero_mask[c]))
    max_range = _central3_slices(value_in_mask)
    return max_range


def central_selector(datapath:str)->str:
    # datapath: os.path.join(dirpath, file) -- data_root + subnaem + filename = 'Root/EAC_Wrist_TRA/Names'
    data = sitk.ReadImage(datapath)
    data_array = sitk.GetArrayFromImage(data)  # output [slice, 512, 512]
    data_array = _normalization(data_array)  # 可以使用class内的normalization
    max_range = _create_nonzero_mask(data_array)  # [slice, 512, 512] the mask for each slice
    return max_range  # 返回一个范围的上下限，然后用':10to15'来保存