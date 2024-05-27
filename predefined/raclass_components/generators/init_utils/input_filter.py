

def input_filter(common_cs_dict:dict, target_category:list, target_site:list, target_dirc:list)->dict:
    keyname = common_cs_dict.keys()  # ['EAC_XXX_XXX', 'CSA_XXX_XXX']
    input_limit = []
    for cate in target_category:
        for site in target_site:
            for dirc in target_dirc:
                input_limit.append(f'{cate}_{site}_{dirc}')
    
    for name in list(keyname):
        if name in input_limit:
            continue
        else:
            del common_cs_dict[name]
    return common_cs_dict