from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import pickle
import pandas as pd
import pickle
# import cv2
import os
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import cv2


def csv_trans(dataset_dir, s):
    df = pd.read_csv(f'{dataset_dir}/csv/{s}')
    df = df[['patient_id','left or right breast','image view','pathology']]  # 取一部分列
    df = df.rename(columns = {'left or right breast':'laterality','image view':'view','pathology':'BIRADS'}) # 改列名
    #sum(df['BIRADS'] != 'BENIGN'),df.shape
    df['BIRADS'] = df['BIRADS'].replace({'BENIGN':0,'BENIGN_WITHOUT_CALLBACK':1,'MALIGNANT':1}) # 更改数据形式
    df['cancer'] = df['BIRADS']
    df.loc[df.cancer == 1,'cancer'] = 1
    df.loc[df.cancer == 0,'cancer'] = 0
    return df


def split_patient_message(s:str):
    parts = s.split('_')
    patient_id = 'P_{}'.format(parts[2])
    return pd.Series([parts[0],patient_id,parts[3],parts[4]])


def remove_nan(image_list, roi_list, label_list, dir:str='D:/ImageNet'):
    assert len(image_list) == len(roi_list)
    assert len(image_list) == len(roi_list)
    return_list = []  # [id[image_path, roi_path, label], ...]
    for i in range(len(image_list)):
        if os.path.exists(os.path.join(dir, image_list[i])) and os.path.exists(os.path.join(dir, roi_list[i])):
            return_list.append([os.path.join(dir, image_list[i]), os.path.join(dir, roi_list[i]), label_list[i]])
    print('length of data:', len(return_list))
    return return_list


def ddsm_initialization(train_dir:str='D:/ImageNet', load_save:bool=False):
    if load_save:
        with open(f'{train_dir}/CBIS-DDSM/path_list.pkl', "rb") as tf:
            stack = pickle.load(tf)
    else:
        core_path = f'{train_dir}/CBIS-DDSM'
        df = pd.read_csv(f'{core_path}/csv/dicom_info.csv')
        # print(df.head())

        df_full_image = df[df.SeriesDescription == 'full mammogram images'].reset_index(drop=True)
        ROI_mask_images = df[df.SeriesDescription == 'ROI mask images'].reset_index(drop=True)
        
        # split the PatientId info into 4 items, so that the second one can fit those of other files
        df_full_image[['dataset_two','patient_id','laterality','view']] = df_full_image.PatientID.apply(split_patient_message)
        ROI_mask_images[['dataset_two','patient_id','laterality','view']] = ROI_mask_images.PatientID.apply(split_patient_message)

        train_calc_data = csv_trans(core_path, 'calc_case_description_train_set.csv')
        test_calc_data = csv_trans(core_path, 'calc_case_description_test_set.csv')
        train_mass_data = csv_trans(core_path, 'mass_case_description_train_set.csv')
        test_mass_data = csv_trans(core_path, 'mass_case_description_test_set.csv')
        csv_all = pd.concat([train_calc_data, train_mass_data])

        print('number of patients: ', len(csv_all.patient_id.unique()))
        print(f'number of normal: {len(csv_all.cancer==0)}, number of benign: {len(csv_all.cancer==1)}, number of cancer: {len(csv_all.cancer==2)}')
        test_csv_all = pd.concat([test_calc_data, test_mass_data])
        print('number of patients: ', len(test_csv_all.patient_id.unique()))
        print(f'number of normal: {len(test_csv_all.cancer==0)}, number of benign: {len(test_csv_all.cancer == 1)}, number of cancer: {len(test_csv_all.cancer == 2)}')

        # check the patient id:
        csv_patient = csv_all.patient_id
        image_patient = df_full_image.patient_id
        roi_patient = ROI_mask_images.patient_id
        print('csv_patient shape', csv_patient.shape, 'image_patient shape', image_patient.shape, 'roi shape', roi_patient.shape)
        inter = set(csv_patient) & set(image_patient) & set(roi_patient)
        print('common part of ids: ', len(inter))
        
        # check the patient id for test
        test_csv_patient = test_csv_all.patient_id
        print('test csv_patient shape', test_csv_patient.shape)
        test_inter = set(test_csv_patient) & set(image_patient) & set(roi_patient)
        print('common part of test ids: ', len(test_inter))
        # create a copy of image and roi path
        test_full_image = df_full_image.copy()
        test_roi_image = ROI_mask_images.copy()

        # select the common part
        csv_all[csv_all['patient_id'].isin(inter)] #& csv_all['cancer'] == 1]
        df_full_image[df_full_image['patient_id'].isin(inter)]
        ROI_mask_images[ROI_mask_images['patient_id'].isin(inter)]

        # select the common part for test
        test_csv_all[test_csv_all['patient_id'].isin(test_inter)]
        test_full_image[test_full_image['patient_id'].isin(test_inter)]
        test_roi_image[test_roi_image['patient_id'].isin(test_inter)]

        # select useful columns
        image_df = df_full_image[['image_path','dataset_two', 'patient_id', 'laterality', 'view']]
        roi_df = ROI_mask_images[['image_path','dataset_two', 'patient_id', 'laterality', 'view']]
        # change the name
        roi_df = roi_df.rename(columns = {'image_path':'roi_path'})

        # for test
        test_image_df = test_full_image[['image_path','dataset_two', 'patient_id', 'laterality', 'view']]
        test_roi_df = test_roi_image[['image_path','dataset_two', 'patient_id', 'laterality', 'view']]
        test_roi_df = test_roi_df.rename(columns = {'image_path':'roi_path'})

        # merge them
        merged_df = pd.merge(csv_all, image_df, on=['patient_id', 'laterality', 'view'])
        merged_df = pd.merge(merged_df, roi_df, on=['patient_id', 'laterality', 'view'])
        print(merged_df.shape,csv_all.shape,image_df.shape,roi_df.shape)

        # merge for test
        merged_test_df = pd.merge(test_csv_all, test_image_df, on=['patient_id', 'laterality', 'view'])
        merged_test_df = pd.merge(merged_test_df, test_roi_df, on=['patient_id', 'laterality', 'view'])
        print(merged_test_df.shape,test_csv_all.shape,test_image_df.shape,test_roi_df.shape)

        # drop duplicates
        merged_df.drop_duplicates(subset=['image_path'],inplace = True) 
        merged_df.drop_duplicates(subset=['roi_path'],inplace = True) 
        df_less = merged_df.reset_index(drop= True)

        # drop for test
        merged_test_df.drop_duplicates(subset=['image_path'],inplace = True) 
        merged_test_df.drop_duplicates(subset=['roi_path'],inplace = True) 
        df_test_less = merged_test_df.reset_index(drop= True)

        print(df_less.shape, df_test_less.shape)

        # check if the image path exist
        image_path = df_less.image_path
        exits = [os.path.exists(os.path.join(train_dir, path)) for path in image_path]
        print(f'{sum(exits)} paths exits')
        print(f'{len(exits)-sum(exits)} paths do not exits')

        # get the data
        train_image_path_list = df_less.image_path.values
        train_roi_path_list = df_less.roi_path.values
        train_label_list = df_less.cancer.values

        test_image_path_list = df_test_less.image_path.values
        test_roi_path_list = df_test_less.roi_path.values
        test_label_list = df_test_less.cancer.values

        train_list = remove_nan(train_image_path_list, train_roi_path_list, train_label_list, dir=train_dir)
        test_list = remove_nan(test_image_path_list, test_roi_path_list, test_label_list, dir=train_dir)

        stack = [train_list, test_list]  # [[full_img_paths[i], gt_paths[i], labels[i]]*N, []*M]
        with open(f'{train_dir}/CBIS-DDSM/path_list.pkl', "wb") as tf:
            pickle.dump(stack, tf)

    # start from stack = [[full_img_paths[i], gt_paths[i], labels[i]]*N, []*M]
    return stack[0], stack[1]


class DDSMDataset(Dataset):
    def __init__(self, stacked_list:list=None, transform=None, val_flag:bool=False):
        # define transformation
        if val_flag:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize((512, 256))])
        else:
            if transform==None:
                self.transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomRotation(10),
                                                transforms.Resize((512, 256)),
                                                ])
            else:
                self.transform = transform
        # set dataset
        self.paths = stacked_list

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        paths = self.paths[index]
        full_img_path = paths[0]
        # gt_seg = paths[1]
        
        # read image
        image = cv2.imread(full_img_path)
        if self.transform is not None:
            image = self.transform(image)

        label_str = paths[2]
        if label_str == 1:
            label = 1
        elif label_str == 0:
            label = 0
        else:
            raise ValueError(f'Wrong label: {label_str}, type:{type(label_str)}')
        
        return image.float(), label


if __name__ == '__main__':
    a, b = ddsm_initialization(train_dir='D:/ImageNet', load_save=False)