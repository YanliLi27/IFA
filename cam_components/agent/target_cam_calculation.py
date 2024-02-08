import numpy as np


def target_cam_selection(importance_matrix: np.array, mode='max', extra=0.5):
    '''
    importrance_matrix: the numpy array of cam importance -- [num_classes, length_feature]
    mode: optional, available choices: max, top, freq, index, default 'max'
    extra: the attribute for mode. 
            (1) For max, extra is useless attribute
            (2) For top, the ratio of top 'extra'
            (3) For freq, the threshold of freq, above 'extra' will be selected
            (4) For index, the chosen index 'extra'
            (5) For diff_top, the ratio of top 'extra', yet the top extra means the abs of the input
    return the merged importance matrix -- [num_classes, length_feature]
    '''
    assert mode in ['max', 'top', 'diff_top', 'freq', 'index', 'all', 'reverse_diff_top']
    # im [num_classes, num_features]
    ClassLen, FeatureLen = importance_matrix.shape
    Return_dtype = importance_matrix.dtype
    return_im = np.zeros([ClassLen, FeatureLen], dtype=Return_dtype)
    for single_class, item in enumerate(importance_matrix):
        # item array[256]
        if mode=='max':
            max_value = np.max(item)
            item[item<max_value] = 0
            item[item>=max_value] = 1
        elif mode=='freq':
            item[item<=extra] = 0
            item[item>extra] = 1
        elif mode=='top':
            sorted_item = sorted(item)
            extra_index = int(extra * len(sorted_item))
            top_value = sorted_item[len(sorted_item)-extra_index]
            item[item<=top_value] = 0
            item[item>top_value] = 1
        elif mode=='diff_top':
            abs_item = np.abs(item)
            sorted_item = sorted(abs_item)
            extra_index = int(extra * len(sorted_item))
            top_value = sorted_item[len(sorted_item)-extra_index]
            abs_item[abs_item<top_value] = 0
            abs_item[abs_item>=top_value] = 1
            item = abs_item
        elif mode=='reverse_diff_top':
            abs_item = np.abs(item)
            sorted_item = sorted(abs_item)
            extra_index = int(extra * len(sorted_item))
            top_value = sorted_item[len(sorted_item)-extra_index]
            abs_item[abs_item<top_value] = 1
            abs_item[abs_item>=top_value] = 0
            item = abs_item
        elif mode=='index':
            item[:extra] = 0
            item[extra+1:] = 0
            item[extra] = 1
        elif mode=='all':
            item[:] = 1
        else:
            raise ValueError('not support mode')
            
        return_im[single_class] = item
    
    return return_im  # [num_classes, length_feature]

            