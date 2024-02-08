import pandas as pd
import re


def ramris_search(df:pd.DataFrame, id:str, site:str, biomarker:str, reader:str, num_nan:int=0)->list:
    # 读取数据并合并成一个list比较好保存到csv
    ramris_list = []
    df_header = [column for column in df]

    if site == 'Wrist':
        site_name = 'WR'
        length = [15 ,15, 3, 10]
    elif site == 'MCP':
        site_name = 'MC'
        length = [8, 8, 4, 8]
    elif site == 'Foot':
        site_name = 'MT'
        length = [10, 10, 5, 10]
    else:
        raise ValueError(f'not valid site:{site}')
    biomarker_len = {'ERO':0, 'BME':1, 'SYN':2, 'TSY':3}
    
    if reader == 'Reader1':
        reader_name = '1'
    elif reader == 'Reader2':
        reader_name = '2'
    else:
        raise ValueError('not valid reader')
    
    for header in df_header:
        search_result = re.search(f'{site_name}(.*){biomarker}(.*)\.{reader_name}', header)
        if search_result:
            scores = df.loc[int(id)].at[header]
            try:
                if type(scores) == str:
                    scores.replace(' ', '')
                scores = int(scores)
            except:
                scores = num_nan
            ramris_list.append(scores)  # NOT USE INT(df.loc[int(id)].at[header]) because there could be nan value, just keep it str
    assert(len(ramris_list)==length[biomarker_len[biomarker]])
    return ramris_list