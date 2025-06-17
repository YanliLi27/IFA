from typing import Union, Tuple, Literal
import os


def namefinder(name_str:str, select_category:Union[None, str, int, list]=None, cam_method:str='gradcam', 
               rescale:Literal['norm', 'tanh', 'norm_multi', 'tanh_multi', None, False]='norm', remove_minus_flag:bool=False, 
               feature_selection:Literal['reverse_diff_top', 'max', 'top', 'diff_top', 'freq', 'index', 'all']='all', 
               fsr:Union[str, float, None]='1') -> Tuple[str, str, str]:
    
    # IM：保存在./output/task(指定)_fold/im/_type_cate_method.csv 下  #可以在init中直接调用运行
    # 图像：保存至./output/task(指定)_fold/cam/_cate_method/scale_rm_feature/os.path.basename(input_name) 中  # 直到name为止均可以在init中构建
    # 指标记录：保存至./output/task(指定)_fold/metric/_cate_method/scale_rm_feature/name # 直到name为止均可以在init中构建
    im_path = f'./output/{name_str}/im/All_im_cate{str(select_category)}_{cam_method}.csv'
    if not os.path.exists(os.path.dirname(im_path)):
        os.makedirs(os.path.dirname(im_path))
    rescale_bool:bool = True if rescale!=False else False
    feature_selection_bool:bool = False if feature_selection=='all' else True
    cam_dir = f'./output/{name_str}/cam/{cam_method}/scale{str(rescale_bool)}_rm{str(remove_minus_flag)}_feature{str(feature_selection_bool)}{str(fsr)}/cate{str(select_category)}'
    if not os.path.exists(cam_dir):
        os.makedirs(cam_dir)
    record_dir = f'./output/{name_str}/metric/{cam_method}/scale{str(rescale_bool)}_rm{str(remove_minus_flag)}_feature{str(feature_selection_bool)}{str(fsr)}/cate{str(select_category)}'
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)
    return im_path, cam_dir, record_dir
