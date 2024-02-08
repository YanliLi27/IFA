import os
import pandas as pd

def ramris_reader(target_category:str)->pd.DataFrame:
    assert target_category in ['EAC', 'CSA', 'ATL']
    RAMRIS_dir = 'D:\\ESMIRA\\SPSS data'
    RAMRIS_list = ['1. EAC baseline.csv', '3. Atlas.csv', '5. CSA_T1_MRI_scores_SPSS.csv']
    if target_category == 'EAC':
        chosen_ramris = RAMRIS_list[0]
        end_point = 9  # 1-9 is the extra info
    elif target_category == 'CSA':
        chosen_ramris = RAMRIS_list[2]
        end_point = 8
    elif target_category == 'ATL':
        chosen_ramris = RAMRIS_list[1]
        end_point = 3
    else:
        raise ValueError('not valid class')
    RAMRIS_path = os.path.join(RAMRIS_dir, chosen_ramris)
    df = pd.read_csv(RAMRIS_path, sep=';')

    # drop useless info
    df_header = [column for column in df]
    df_index = df_header[0]
    drop_names = df_header[1:end_point+1]
    df.drop(columns=drop_names)
    df = df.set_index(df_index)
    return df
    # ramris_ids = df[df_header[0]].values  # type: np.ndarray
    # ramris_ids_regularized = []

    # for id in ramris_ids:
    #     if target_category == 'EAC':  # Arth3393_EAC
    #         id = 'Arth' + str(id).zfill(4) + '_EAC'
    #         ramris_ids_regularized.append(id)
    #     elif target_category == 'CSA':  # Csa014_CSA
    #         id = 'Csa' + str(id).zfill(3) + '_CSA'
    #         ramris_ids_regularized.append() 
    #     elif target_category == 'ATL':  # Atlas002_ATL
    #         id = 'Atlas' + str(id).zfill(3) + '_ATL'
    #         ramris_ids_regularized.append()
    #     else:
    #         raise ValueError('NoT VALID CLASS')
            