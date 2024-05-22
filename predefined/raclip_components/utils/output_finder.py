import os


def output_finder(target_category:list, target_site:list, target_dirc:list, target_biomarker, fold_order:int, sumscore:bool=False)->str:
    cate_name = ''
    if target_category is not None:
            for cate in target_category:
                cate_name = cate_name + cate + '_'
    else:
        cate_name = cate_name + 'All_'
    if len(target_site) > 1:
        site_name = str(len(target_site)) + 'site'
    else:
        site_name = target_site[0]
    
    if len(target_dirc) <2:
        dirc_name = target_dirc[0]
    else:
        if 'TRA' in target_dirc or 'COR' in target_dirc:
            dirc_name = '2dirc'
        else:
            dirc_name = '2read'
    sumscore_flag = 'Sum' if sumscore else ''
    if target_biomarker:  
        if len(target_biomarker)>1:
            replace = 'ALLBIO'
        else:
            replace = f'ALL{target_biomarker[0]}'
        output_name = "./models/weights/{}/{}_{}_{}_fold{}{}.model".format(replace, cate_name, site_name, dirc_name, fold_order, sumscore_flag)
    else:
        output_name = "./models/weights/{}_{}_{}_fold{}{}.model".format(cate_name, site_name, dirc_name, fold_order, sumscore_flag)
    output_dir = os.path.dirname(output_name)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    return output_name