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

    
    def show_cam_on_image2d(img: np.ndarray,
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
        # origin [channel, L, W]
        cam_image_group_list = []
        for j in range(self.groups):
            cam_group = cam[j, :]  # [channel, L, W] --> [L, W]
            cam_image_group = self.show_cam_on_image2d(origin[j], cam_group, use_rgb=True, use_origin=use_origin)
            cam_image_group = cv2.cvtColor(cam_image_group, cv2.COLOR_RGB2BGR)
            cam_image_group_list.append(cam_image_group)
        
        origin_list = []
        for img in origin:  # img_color_group [batch, organ_groups, y, x, 3] 
            origin_list.append(img * 255)  # img - [L, W]

        concat_img_origin = cv2.hconcat(origin_list) # [x, N*y, 3]

        concat_img_cam = cv2.hconcat(cam_image_group_list) # [3, x, N*y]
        concat_img_cam = concat_img_cam.astype(concat_img_origin.dtype)
        concat_img_all = cv2.vconcat([concat_img_cam, concat_img_origin])
        return concat_img_all
        

    def img_create(self, cam:np.array, origin:np.array, tc, name_str:str, use_origin:bool=True) ->None:
        # name_str mainly used (indiv info, group, dirc included), name extra for the situation that one origin have multiple cam
        
        # origin [channel(groups*n), (D), L, W] 
        # channel exists only when D doesnt, group can exist when multiinput using group conv
        origin_shape = origin.shape
        # cam [groups, (D), L, W]
        cam_shape = cam.shape

        # 四种情况：双3D出3D, \\ 双3D出2D, 双2D出2D, 3D+2D出2D根据groups来选
        if self.cam_type == '2D':
            if len(cam_shape)==4 and len(origin_shape)==4:
                if not cam_shape==origin_shape:
                    raise ValueError(f'the shape of the cam {cam_shape} should match that of origin {origin_shape}')
                _, D, _, _ = origin_shape
                for g in range(self.groups):  # [D, L, W]
                    # channel exists only when D doesnt, group can exist when multiinput using group conv
                    for d in range(D):
                        concat_img_all = self.img_creator_2D(cam[g, d, :], origin[g, d, :], name_str, use_origin)
                        save_name = os.path.join(self.cam_dir[str(tc)], f'{name_str}_g{g}_s{d}.jpg')
                        cv2.imwrite(save_name, concat_img_all)
                        if self.backup:
                            backup_name = save_name.replace('.jpg', '.npy')
                            np.save(backup_name, np.asarray({'img':origin[g, d, :],'cam':cam[g, d, :]}))
            elif len(cam_shape)==3 and len(origin_shape)==4:
                _, D, _, _ = origin_shape
                origin = origin[:, D//2, :]  # select a slice
                for g in range(self.groups):
                # cam2d and origin2d [groups, L, W]
                    concat_img_all = self.img_creator_2D(cam[g, :], origin[g, :], name_str, use_origin)
                    save_name = os.path.join(self.cam_dir[str(tc)], f'{name_str}_g{g}.jpg')
                    cv2.imwrite(save_name, concat_img_all)
                    if self.backup:
                        backup_name = save_name.replace('.jpg', '.npy')
                        np.save(backup_name, np.asarray({'img':origin[g, :],'cam':cam[g, :]}))
            elif len(cam_shape)==3 and len(origin_shape)==3:
                for g in range(self.groups):
                # cam2d and origin2d [groups, L, W]
                    concat_img_all = self.img_creator_2D(cam[g, :], origin[g, :], name_str, use_origin)
                    save_name = os.path.join(self.cam_dir[str(tc)], f'{name_str}_g{g}.jpg')
                    cv2.imwrite(save_name, concat_img_all)
                    if self.backup:
                        backup_name = save_name.replace('.jpg', '.npy')
                        np.save(backup_name, np.asarray({'img':origin[g, :],'cam':cam[g, :]}))

        elif self.cam_type == '3D':
            if not (len(origin_shape)==4 and len(cam_shape)==4):
                raise ValueError(f'shape of origin {origin_shape} and cam {cam_shape} doesnt meet the requirement of 3D output')
            for g in range(self.groups):
                save_name = os.path.join(self.cam_dir[str(tc)], f'{name_str}_g{self.groups}.nii.gz')
                writter = sitk.ImageFileWriter()
                writter.SetFileName(save_name)
                writter.Execute(sitk.GetImageFromArray(cam[g]))

                origin_save_name = os.path.join(self.cam_dir[str(tc)], f'{name_str}_origin_g{self.groups}.nii.gz')
                if not os.path.isfile(origin_save_name):
                    writter.SetFileName(origin_save_name)
                    # [batch, organ_groups, z, y, x, channel] to [batch, organ_groups, z, y, x]
                    writter.Execute(sitk.GetImageFromArray(origin[g]))

        elif self.cam_type == '1D':
            raise AttributeError('1D not supported yet')
        