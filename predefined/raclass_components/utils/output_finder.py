import os


def output_finder(model_name:str, target_category:list, target_site:list, target_dirc:list, fold_order:int)->str:
    cate_name = ''
    for cate in target_category:
        cate_name = cate_name + cate + '_'
    if len(target_site) > 1:
        site_name = str(len(target_site)) + 'site'
    else:
        site_name = target_site[0]
    
    if len(target_dirc) <2:
        dirc_name = target_dirc[0]
    else:
        dirc_name = '2dirc'
    
    output_name = "./models/weights/{}/{}_{}_{}_fold{}.model".format(model_name, cate_name, site_name, dirc_name, fold_order)
    output_dir = os.path.dirname(output_name)
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    return output_name