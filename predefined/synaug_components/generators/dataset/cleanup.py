from scipy.ndimage import binary_fill_holes, binary_opening, binary_closing   # 填充
import numpy as np


def _create_nonzero_mask_for_clean(data):  # data: [20, 512, 512]
    
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[:], dtype=bool)  # [512, 512] 因为20层应该是要全用，所以说完全可以把z轴作为channel
    for c in range(data.shape[0]):  # 读取单个sample
        max_value = np.max(data[c])
        this_mask = data[c] >= 0.1 * max_value # threshold = 0.1,  out of range(0, 1) after normalization
        nonzero_mask[c] = nonzero_mask[c] | this_mask  # 有一个为1则为1， 在循环后相当于取各个slice的最大值
    for c in range(len(nonzero_mask)):
        nonzero_mask[c] = binary_fill_holes(nonzero_mask[c])
        # 此后进行开闭操作去除artifacts： 开操作先腐蚀去除奇异点，再扩张还原。 闭操作使边缘均匀化，未必需要。
        nonzero_mask[c] = binary_opening(nonzero_mask[c], iterations=20)
        nonzero_mask[c] = binary_closing(nonzero_mask[c], iterations=1)
    return nonzero_mask


def clean_main(data_array:np.array) -> np.array:
    nonzero_mask = _create_nonzero_mask_for_clean(data_array)  # [slice, 512, 512] the mask for each slice
    data_array[nonzero_mask == 0] = 0
    return data_array