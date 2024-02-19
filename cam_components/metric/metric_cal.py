import numpy as np
from typing import Union
import os
from scipy import stats
from sklearn.metrics import roc_auc_score, accuracy_score
from cam_components.metric.eval_utils import *
import torch
import torch.nn as nn



class EvalAgent():
    def __init__(self, save_path:dict, eval_act:Union[bool, str]=False, creator_tc:list=[None], num_classes:int=2, groups:int=1) -> None:
        assert eval_act in [False, 'false', 'basic', 'logit', 'corr', 'corr_logit']
        # eval - corr #  以下这些需要选定一个特定的类别
        self.save_path:dict = save_path  # tc:path
        self.eval_func = eval_act
        self.num_classes:int = num_classes 
        self.groups:int = groups
        self.counter = {}   # for global calculation
        self.corr_cam_matrix = {}  # tc: cam_sum_value for each sample -- an array
        self.corr_pred_matrix = {}  # tc: pred_class for each sample -- an array  # 计算二值化的AUC
        self.corr_conf_matrix = {}  # tc: logit/confidence_value for each sample -- an array  # 计算
        if eval_act in ['corr', 'logit']:
            for i in range(len(creator_tc)):
                if not isinstance(creator_tc[i], int):
                    raise AssertionError('the select category should be integers while conducting the correlation-based evaluation')
        # eval - block #  以下这些都需要在tc=None的情况下才能有用
        self.increase = {}  # tc: int
        self.drop = {}  # tc: int  
        self.label_gt = {}  # tc: y
        self.label_origin = {}  # tc: pred  -- an array
        self.label_cam = {}  # tc: cam_pred -- an array
        self.label_fm = {}  # tc: fm_pred -- an array
        if eval_act in ['logit', 'basic']:
            if creator_tc[0] != None:
                raise AssertionError('the select category should be None while conducting the blocking-based evaluation')
        
        # eval - init #
        list_metrics = [self.corr_cam_matrix, self.corr_pred_matrix, self.corr_conf_matrix, 
                   self.label_gt, self.label_origin, self.label_cam, self.label_fm]
        num_metrics = [self.increase, self.drop, self.counter]
        for metric in list_metrics:
            for tc in creator_tc:
                metric[str(tc)] = []
        for metric in num_metrics:
            for tc in creator_tc:
                metric[str(tc)] = 0.0


    def eval(self, tc, grayscale_cam:np.array, predict_category, pred_score):
        # 3d grayscale cam -- 16 * [1, 5, 256, 256] - batch * [1, 5, 256, 256]
        # batch_size * [group_num, (z,) y, x]
        for i, single_cam in enumerate(grayscale_cam):  # 取单个进行计算和存储
            # [batch, groups, (D), L, W] -> [groups, (D), L, W]
            self.corr_cam_matrix[str(tc)].append(np.sum(single_cam))  # tc: cam_sum_value for each sample -- an array
            self.corr_pred_matrix[str(tc)].append(predict_category[i])    # tc: pred_class for each sample -- an array  # 计算二值化的AUC
            self.corr_conf_matrix[str(tc)].append(pred_score[i]) 
            self.counter[str(tc)] += 1
    

    def blockeval(self, tc, grayscale_cam:np.array, predict_category, pred_score, nega_score, 
                  x:torch.Tensor, y:torch.Tensor, model:nn.Module,
                  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        # choose from one: cam_regularizer or cam_regularizer_binary
        grayscale_cam = cam_regularizer(np.array(grayscale_cam)) # -- [16, 1, 256, 256]
        # grayscale_cam = cam_regularizer_binary(np.array(grayscale_cam)) # -- [16, 1, 256, 256]
        # grayscale_cam:numpy [batch, groups, length, width] from 0 to 1, x:tensor [batch, in_channel, length, width] from low to high
        extended_cam = np.zeros(x.shape, dtype=np.float32)
        channel_per_group = x.shape[1] // self.groups
        for gc in range(self.groups):
            extended_cam[:, gc*channel_per_group:(gc+1)*channel_per_group, :] = np.expand_dims(grayscale_cam[:, gc, :], axis=1)
        # extended_cam: numpy [batch, in_channel, length, width]
        cam_input = torch.from_numpy(extended_cam).to(device) * x
        cam_input = cam_input_normalization(cam_input)
        cam_pred = model(cam_input)
        if self.eval_func == 'basic':
            origin_category, single_origin_confidence = predict_category, pred_score
            _, single_cam_confidence, _ = pred_score_calculator(x.shape[0], cam_pred, tc,
                                                                        origin_pred_category=origin_category,
                                                                        out_logit=False)
            single_drop = torch.relu(torch.from_numpy(single_origin_confidence\
                        - single_cam_confidence)).div(torch.from_numpy(single_origin_confidence) + 1e-7)
        elif self.eval_func == 'logit':
            origin_category, single_origin_confidence = y, pred_score
            _, single_cam_confidence, single_cam_nega_scores = pred_score_calculator(x.shape[0], cam_pred, 'GT',
                                                                                origin_pred_category=origin_category)
            
            single_drop = nega_score > single_cam_nega_scores  # 新的drop越大越好
        self.label_cam[str(tc)].extend(np.argmax(softmax(cam_pred.cpu().data.numpy(), axis=-1),axis=-1))
        self.label_origin[str(tc)].extend(predict_category)
        self.label_gt[str(tc)].extend(y.cpu().data.numpy())
        self.counter[str(tc)] += x.shape[0]
        single_increase = single_origin_confidence < single_cam_confidence
        self.increase[str(tc)] += single_increase.sum().item()
        self.drop[str(tc)] += single_drop.sum().item()


    def evalsummary(self, tc):  # 根据累积的数值进行计算
        # --------------------------------------  cam evaluate  -------------------------------------- #
        if self.eval_func in ['corr', 'corr_logit']:
            print('total samples:', self.counter[str(tc)])
            # cam分数和类别的AUROC，代表的是cam正确反映分类情况的能力
            # for mutliclasses, use pos-neg to calculate
            if self.num_classes>2:
                reg_corr_pred_matrix = []
                for item in self.corr_pred_matrix[str(tc)]:
                    if item==tc:
                        reg_corr_pred_matrix.append(1)
                    else:
                        reg_corr_pred_matrix.append(0)
                corr_pred_matrix = np.asarray(reg_corr_pred_matrix)
            else:
                corr_pred_matrix = np.nan_to_num(self.corr_pred_matrix[str(tc)], copy=False, nan=0, posinf=0, neginf=0)
            corr_cam_matrix = np.nan_to_num(self.corr_cam_matrix[str(tc)], copy=False, nan=0, posinf=0, neginf=0)
            auc = roc_auc_score(corr_pred_matrix, corr_cam_matrix)
            print('outlier rate-- AUROC of <CAM & Label>: ', auc)
            save_dir = self.save_path[str(tc)]
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_name = os.path.join(save_dir, f'or_scatter_{str(auc)[:5]}.jpg')
            scatter_plot(corr_pred_matrix, corr_cam_matrix, fit=False, save_path=save_name)
            print(f'or scatter plot saved: {save_name}')
            
            # cam分数与pred的corr，代表CAM正确反映pred的能力，也即是weight与真实重要程度的关系情况
            corr, p_value = stats.pearsonr(corr_cam_matrix, self.corr_conf_matrix[str(tc)])       
            
            print('corrlation of <CAM & Pred scores>: ', corr)
            print('p value: ', p_value)
            corr_save_name = os.path.join(save_dir, f'corr_scatter_{str(corr)[:6]}_{str(p_value)[-6:]}.jpg')
            scatter_plot(self.corr_conf_matrix[str(tc)], corr_cam_matrix, save_path=corr_save_name)
            print(f'corr scatter plot saved: {corr_save_name}')

        elif self.eval_func in ['basic', 'logit']:
            print('total samples:', self.counter[str(tc)])
            avg_increase = self.increase[str(tc)]/self.counter[str(tc)]
            avg_drop = self.drop[str(tc)]/self.counter[str(tc)]

            acc_original = accuracy_score(self.label_gt[str(tc)], self.label_origin[str(tc)])
            acc_cammasked = accuracy_score(self.label_gt[str(tc)], self.label_cam[str(tc)])
            # acc_original = roc_auc_score(acc_gt, acc_ori)
            # acc_cammasked = roc_auc_score(acc_gt, acc_cam)

            print('increase:', avg_increase)
            print('avg_drop:', avg_drop)
            print('acc of original:', acc_original)
            print('acc after masking:', acc_cammasked)
            save_dir = self.save_path[str(tc)]
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            eval_borl_save_name = os.path.join(save_dir, f'eval_with_{self.eval_func}.txt')
            text_save(eval_borl_save_name, avg_increase, avg_drop, self.counter[str(tc)])
            text_save_acc(os.path.join(save_dir, 'eval_with_acc.txt'), acc_original, acc_cammasked, self.counter[str(tc)])