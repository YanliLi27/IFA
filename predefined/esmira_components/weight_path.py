def output_finder(target_category:list, target_site:list, target_dirc:list, fold_order:int)->str:
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
    
    output_name = "{}_{}_{}_fold{}.model".format(cate_name, site_name, dirc_name, fold_order)
    return output_name