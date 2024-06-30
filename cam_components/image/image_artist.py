from typing import Union
import cv2
import numpy as np
import os
import SimpleITK as sitk


class Artists:
    def __init__(self, cam_dir:dict, cam_type:Union[str, None], groups:int=1, backup:bool=True) -> None:
        if not cam_type:
            cam_type = '2D'
        self.cam_type = cam_type
        self.groups = groups
        self.cam_dir = cam_dir  
        # {'tc':f'./output/{name_str}/cam/cate{str(select_category)}_{cam_method}/scale{str(rescale)}_rm{str(remove_minus_flag)}_feature{str(feature_selection)}{kwarg}'}
        self.backup = backup

        # 需要一个process origin的功能，确保尽可能简单

    
    def show_cam_on_image2d(self,
                            img: np.ndarray,
                            mask: np.ndarray,
                            use_rgb: bool = False,
                            colormap: int = cv2.COLORMAP_JET,
                            use_origin: bool = True,
                            ) -> np.ndarray:
                            # colormap: int = cv2.COLORMAP_BONE) -> np.ndarray:
        # colormap can be searched through cv2.COLORMAP + _
        """ This function overlays the cam mask on the image as an heatmap.
        By default the heatmap is in BGR format.

        :param img: The base image in RGB or BGR format.
        :param mask: The cam mask.
        :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
        :param colormap: The OpenCV colormap to be used.
        :param use_origin: if overlay them
        :returns: The default image with the cam overlay.
        """
        mask = np.maximum(mask, 0)
        mask = np.minimum(mask, 1)
        if len(mask.shape) > 2:
            heatmap = np.uint8(255 * mask)
        else:
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
        if use_rgb:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        # heatmap = np.float32(heatmap) / 255

        # if np.max(img) > 1:
        #     raise Exception(
        #         "The input image should np.float32 in the range [0, 1]")
        # if use_origin:
        #     cam = heatmap + img
        # else:
        #     cam = heatmap
        # cam = cam / np.max(cam)
        # return np.uint8(255 * cam)
        heatmap = np.float32(heatmap)

        if np.max(img) > 1:
            raise Exception(
                "The input image should np.float32 in the range [0, 1]")
        img = (img * 255).astype(np.float32)
        if use_origin:
            cam = cv2.addWeighted(heatmap, 0.7, img, 0.3, 0)
        else:
            cam = heatmap

        return np.uint8(cam)

    
    def img_creator_2D(self, cam, origin, use_origin:bool=True):
        # origin [self.groups, channel, L, W]  # cam [self.groups, L, W]
        origin = origin - np.min(origin)
        origin = origin / (np.max(origin) +1e-7)
        if origin.shape[1] == 1:
            origin = np.repeat(origin, repeats=3, axis=1)  # [G, 1, L, W] -> [G, 3, L, W]
        elif origin.shape[1]!=3:
            origin = np.repeat(np.expand_dims(origin[:, origin.shape[1]//2, :], axis=1) , repeats=3, axis=1)
        origin = np.transpose(origin, axes=(0, 2, 3, 1))  # [G, L, W, 3]
        cam_image_group_list = []
        for j in range(self.groups):
            # cam [self.groups, L, W] --> [L, W], origin [self.groups, channel, L, W] --> [channel, L, W]
            cam_image_group = self.show_cam_on_image2d(origin[j], cam[j], use_rgb=True, use_origin=use_origin)
            cam_image_group = cv2.cvtColor(cam_image_group, cv2.COLOR_RGB2BGR)
            cam_image_group_list.append(cam_image_group)
        
        origin_list = []
        for j in range(self.groups):  # img_color_group [batch, organ_groups, y, x, 3] 
            origin_list.append(origin[j] * 255)  # img - [C, L, W]

        concat_img_origin = cv2.hconcat(origin_list) # [x, N*y, 3]

        concat_img_cam = cv2.hconcat(cam_image_group_list) # [3, x, N*y]
        concat_img_cam = concat_img_cam.astype(concat_img_origin.dtype)
        concat_img_all = cv2.vconcat([concat_img_cam, concat_img_origin])
        return concat_img_all
        

    def img_create(self, cam:np.array, origin:np.array, name_str:str, use_origin:bool=True) ->None:
        # name_str mainly used (indiv info, group, dirc included), name extra for the situation that one origin have multiple cam
        
        # origin [channel(groups*n), (D), L, W] 
        # channel exists only when D doesnt, group can exist when multiinput using group conv
        origin_shape = origin.shape
        if len(origin.shape)==3: 
            origin = np.reshape(origin, (self.groups, -1, origin_shape[-2], origin_shape[-1]))
        elif len(origin.shape)==4:
            origin = np.reshape(origin, (self.groups, -1, origin_shape[-3], origin_shape[-2], origin_shape[-1]))
        origin_shape = origin.shape
        # cam [groups, (D), L, W]
        cam_shape = cam.shape
        # 四种情况：双3D出3D, \\ 双3D出2D, 双2D出2D, 3D+2D出2D根据groups来选
        if self.cam_type == '2D':
            if len(cam_shape)==4 and len(origin_shape)==5:
                if not cam_shape[-3:]==origin_shape[-3:]:
                    raise ValueError(f'the shape of the cam {cam_shape} should match that of origin {origin_shape}')
                _, D, _, _, _ = origin_shape
                for d in range(D):
                    concat_img_all = self.img_creator_2D(cam[:, d, :], origin[:, :, d, :], use_origin)
                    # [G, D, L, W] -> [G, L, W], [G, C, D, L, W] -> [G, C, L, W]
                    save_name = f'{name_str}_s{d}.jpg'
                    cv2.imwrite(save_name, concat_img_all)
                    if self.backup:
                        backup_name = save_name.replace('.jpg', '.npy')
                        np.save(backup_name, np.asarray({'img':origin[:, d, :],'cam':cam[:, :, d, :]}))
            elif len(cam_shape)==3 and len(origin.shape)==4:
                concat_img_all = self.img_creator_2D(cam, origin, use_origin)
                # [G, L, W] , [G, C, L, W]
                save_name = f'{name_str}.jpg'
                cv2.imwrite(save_name, concat_img_all)
                if self.backup:
                    backup_name = save_name.replace('.jpg', '.npy')
                    np.save(backup_name, np.asarray({'img':origin,'cam':cam}))
            elif len(cam_shape)==2 and len(origin.shape)==3:
                concat_img_all = self.img_creator_2D(cam, origin, use_origin)
                # [G, W] , [G, C, W]
                save_name = f'{name_str}.jpg'
                cv2.imwrite(save_name, concat_img_all)
                if self.backup:
                    backup_name = save_name.replace('.jpg', '.npy')
                    np.save(backup_name, np.asarray({'img':origin,'cam':cam}))

        elif self.cam_type == '3D':
            if not (len(origin_shape)==5 and len(cam_shape)==4):
                raise ValueError(f'shape of origin {origin_shape} and cam {cam_shape} doesnt meet the requirement of 3D output')
            for g in range(self.groups):
                save_name = os.path.join(f'{name_str}_g{g}.nii.gz')
                writter = sitk.ImageFileWriter()
                writter.SetFileName(save_name)
                writter.Execute(sitk.GetImageFromArray(cam[g]))

                origin_save_name = os.path.join(f'{name_str}_origin_g{g}.nii.gz')
                if not os.path.isfile(origin_save_name):
                    writter.SetFileName(origin_save_name)
                    # [batch, organ_groups, z, y, x, channel] to [batch, organ_groups, z, y, x]
                    writter.Execute(sitk.GetImageFromArray(origin[g, 0]))

        elif self.cam_type == '1D':
            raise AttributeError('1D not supported yet')
        