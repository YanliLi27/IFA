def split_saver(target_category:list, target_site:list, target_dirc:list, target_flag:bool=True)->str:
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
    if target_flag:
        target_name = 1
    else:
        target_name = 0
    output_name = "./predefined/esmira_components/dataset/logs/{}_{}_{}_{}.pkl".format(cate_name, site_name, dirc_name, target_name)
    return output_name