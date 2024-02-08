import numpy as np
from typing import Union, Tuple


def resampler(img_list:list, ramris_list:list, mse_list:list, groups:int=10) -> Tuple[list, list, Union[None, list]]:
    # ASSERT IT'S ALREADY SEPARATED DATASET
    # img_list: # [batch, site, path_name], ramris_lsit: # [batch, site, scores], agree_list: # [batch, site, scores]

    # input the img_list and ramris_list, create a list of index for img_list and ramris_list
    # divide ramris_list into several groups, count the number of samples in each group -- save in a counter list
    # using the counter list like the previous balance method, calculate the Num.
    # create an extra list of index, read through the whole ramris_list:
        # add the index * Num to the extra list (it's already separated dataset -- no validation/train mixed)
    # Read through img_list and ramris_list:
        # create an extra img_list and an extra ramris_list
        #  (for item in extra_list of index, extra img_list/ramris_list append img_list[item]/ramris_list[item])
        # img_list + extra img_list, ramris_list +ramris_list
    # return new img_list, ramris_list
    sum_ramris = []
    for item in ramris_list:
        list_data = []
        for data in item:
            list_data.extend(data)
        sum = np.sum(np.asarray(list_data, dtype=np.float32))
        sum_ramris.append(sum)  # sum_ramris 中按照原index保存了ramris的合值
    # 统计元素个数
    hist, bins = np.histogram(np.asarray(sum_ramris), bins=groups, range=(0, groups))
    # hist -- frequency of values in each bin, list : [freq1, freq2, freq3, ...]
    # bins -- list: [division1, ...]
    # create the ratio of aug: np.max(hist)/current_freq
    hist = np.asarray(hist) + 1
    ratio_list = np.max(hist)//hist  # the list of aug ratio
    ratio_list = np.minimum(ratio_list, 10)
    # repete
    extra_img_list = []
    extra_ramris_list = []
    extra_mse_list = []
    assert len(img_list) == len(ramris_list)
    assert len(img_list) == len(sum_ramris)
    assert len(mse_list) == len(img_list)
    for index in range(len(img_list)):
        repeat_index = 0
        if int(sum_ramris[index]) >= 10:
            repeat_index = 9
        elif int(sum_ramris[index]) <=0:
            repeat_index = 0
        else:
            repeat_index = int(sum_ramris[index])
        repeat = ratio_list[repeat_index]

        for _ in range(repeat):
            extra_img_list.append(img_list[index])
            extra_ramris_list.append(ramris_list[index])
            extra_mse_list.append(mse_list[index])

    print('img and ramris balanced')
    return extra_img_list, extra_ramris_list, extra_mse_list
