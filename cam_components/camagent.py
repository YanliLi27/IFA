# 核心是把normalization(max,min,removeminus0)和各种复杂参数挪出来
# 其次是将生成图像这一步通过外部给入参数来进行简化，合并各类名称构建函数，降低整体代码长度
# 评价指标这一运算在这种设计下应该可以同步进行简化
import numpy as np
from typing import Union, Literal
from torch.utils.data import DataLoader, Dataset
from cam_components.methods import GradCAM, FullCAM, GradCAMPP, XGradCAM, ScoreCAM
from cam_components.core.rescale import Rescaler
from cam_components.preparation import *
from cam_components.agent import *
from cam_components.image.image_artist import Artists
from cam_components.metric.metric_cal import EvalAgent
import os
import torch
import platform
from tqdm import tqdm


class CAMAgent():
    def __init__(self,
                model, target_layer, dataset:Union[DataLoader, np.array],  
                groups:int=1, ram:bool=False,
                # ---------------- model and dataset -------------#
                cam_method:str='gradcam', 
                name_str:str='task_fold',  # output setting
                batch_size:int=1, select_category:Union[None, str, int, list]=None,  # info of the running process
                # ---------------- cam setting -------------#
                rescale:Literal['norm', 'tanh', 'norm_multi', 'tanh_multi', None, False]='norm', 
                remove_minus_flag:bool=False, scale_ratio:float=1.,  # rescaler
                feature_selection:Literal['reverse_diff_top', 'max', 'top', 'diff_top', 'freq', 'index', 'all']='all', 
                feature_selection_ratio:Union[float, None]=1.,  # feature selection
                randomization:Union[None, float]=None,  # model randomization for sanity check
                use_pred:bool=False,  # improve effeciency
                rescaler=None,  # outer scaler
                cam_type:Literal['1D', '2D', '3D', None]='2D',  # cam output type
                reshape_transform=None
                ) -> None:
       
        # agent 有如下部分组成：
        # J 0. 命名组件，根据使用的scale，rm0， 图像生成器，样本序数，outputclass等生成路径名称
        #       尽可能简化：共三种输出：IM， 图像和指标记录
        #       IM：保存在./output/task(指定)_fold/im/_type_cate_method.csv 下  #可以在init中直接调用运行
        #       图像：保存至./output/task(指定)_fold/cam/_cate_method/scale_rm_feature/os.path.basename(input_name) 中  # 直到name为止均可以在init中构建
        #       指标记录：保存至./output/task(指定)_fold/metric/_cate_method/scale_rm_feature/name # 直到name为止均可以在init中构建

        # J 1. rescale相关组件，读取IM提供的maxmin等（可能涉及选择percentile），对返回的数据进行rescale。与analyzer（仅输出）和creator均有交互
        #       核心机要在于两点，scale的上下限与运行模式，以及scale的应用范围（不同ram output是否一起apply）

        # 2. 图像生成相关组件，根据输入的图像进行生成，只需要考虑产生的结果的维度。仅返回生成的图像格式，保存应在creator中
        #       需要有一个自主判断图像格式并进行default生成的函数，提高易用性 （维度，彩色等）

        # 3. RAM 启动器，回归任务的IM计算需要遍历所有输出（不能用最大，因为每个输出都有意义）；CAM生成又需要考虑先宽度还是先深度的问题；
        #       主要是会影响循环和图像生成的过程

        # J 4. analyzer （循环设定开始（是否RAM），输入，累计结果保存文件，并传入给scale组件必要的max和min）
        # 5. creator （循环设定开始（是否RAM），输入图片、label得到未经scale的结果，使用scale（采用何种scale（包括是否群体一起scale、ram）），
        #              使用图像生成组件/（计算指标），获得图像和指标并根据命名组件进行保存）

        # cam info
        cam_method_zoo = {"gradcam": GradCAM, 
                        "fullcam": FullCAM,
                        "gradcampp":GradCAMPP,
                        "xgradcam":XGradCAM,
                        "scorecam":ScoreCAM}
        self.cam_method_name = cam_method  # 用于命名文件
        self.cam_method = cam_method_zoo[cam_method]
        self.batch_size = batch_size  # for cam efficiency
        assert (select_category in ['GT', None] or type(select_category) == int or type(select_category)==list)
        self.select_category = select_category  # targeted category

        # model info
        self.model = model
        self.target_layer = target_layer
        # model randomization for sanity check
        if isinstance(randomization, float):
            model_randomizer(self.model, severity=randomization)

        # dataset info
        self.dataset = dataset
        self.ram = ram
        if self.ram:
            print('please notice, for regression tasks, the target categories is necessary, for both analyzer and creator.\
                  if no predefined category, the default is 0.')
        self.groups = groups  # group convolution

        # num_classes, for im creation and metrics calculation
        self.num_classes = model_out_reader(self.model, self.dataset)

        # for analyzer and creator
        self.rescale = rescale
        assert self.rescale in ['norm', 'tanh', 'norm_multi', 'tanh_multi', None, False]
        self.rm = remove_minus_flag
        assert feature_selection in ['reverse_diff_top', 'max', 'top', 'diff_top', 'freq', 'index', 'all']
        self.fs = feature_selection
        self.fsr = feature_selection_ratio if isinstance(feature_selection_ratio, float) and self.fs else 1.0

        self.use_pred = use_pred  # whether improve the effficiency through use the prediction only, assert the existence of the cateNone file

        # ------------------------------------------------------- cam core open ------------------------------------------------------- #
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cuda_flag = True if self.device=='cuda' else False
        self.camoperator = self.cam_method(model=model,
                                 target_layers=self.target_layer,
                                 ram=self.ram,
                                 use_cuda=cuda_flag,
                                 reshape_transform=reshape_transform,
                                 groups=self.groups, # if use group conv, need to seperate them
                                 importance_matrix=None,  # overwrite when using creator, initialize to avoid passing attributes
                                 out_logit=False,  # overwrite when using creator, initialize to avoid passing attributes
                                 )
        # ------------------------------------------------------- cam core open ------------------------------------------------------- #

        # ------------------------------------------------------- get path in dict ------------------------------------------------------- #
        self.im_path = {}
        self.cam_dir = {}
        self.record_dir = {}

        if isinstance(select_category, list):
            for sc in select_category:
                self.im_path[str(sc)], self.cam_dir[str(sc)], self.record_dir[str(sc)] = namefinder(name_str, sc, cam_method, 
                                                                                                    rescale, remove_minus_flag, 
                                                                                                    feature_selection, fsr=feature_selection_ratio)
        else:
            sc = select_category
            self.im_path[str(sc)], self.cam_dir[str(sc)], self.record_dir[str(sc)] = namefinder(name_str, sc, cam_method, 
                                                                                                rescale, remove_minus_flag, 
                                                                                                feature_selection, fsr=feature_selection_ratio)
            # self.im_path['1'] = ./output/name_str/im/All_cate_method.csv
            # self.cam_dir['1'] = ./output/name_str/cam/cate_method/scale_rm_feature/      os.path.basename(input_name)
            # self.record_dir['1'] = ./output/name_str/metric/_cate_method/scale_rm_feature/      name
        # ------------------------------------------------------- get path in dict ------------------------------------------------------- #
            
        self.value_max = {}
        self.value_min = {}
        self.im = {}
        # ------------------------------------------------------- get rescaler ------------------------------------------------------- #
        self.rescaler = {}
        if (self.rescale) and (rescaler is None):
            # without outer input
            if not isinstance(select_category, list):
                select_category = [select_category]
            for tc in select_category:
                im_path = self.im_path[str(tc)]
                if not os.path.exists(im_path):
                    print('calculate the common scale and importance matrix')
                    self._analyzer_main()
            for tc in select_category:
                self.im[str(tc)], self.value_max[str(tc)], self.value_min[str(tc)] = self._get_importances(im_path, tc)

            if 'multi' in self.rescale:
                value_max = max(self.value_max.values)
                value_min = min(self.value_min.values)
                self.rescaler['uniform'] = Rescaler(value_max=value_max, value_min=value_min, remove_minus_flag=self.rm,
                                                     rescale_func=rescale, scale_ratio=scale_ratio)
            else:
                for tc in select_category:
                    self.rescaler[str(tc)] = Rescaler(value_max=self.value_max[str(tc)], value_min=self.value_min[str(tc)], 
                                                      remove_minus_flag=self.rm, rescale_func=rescale, scale_ratio=scale_ratio)
                
        else:
            self.rescaler['uniform'] = Rescaler(value_max=None, value_min=None, remove_minus_flag=self.rm, rescale_func='norm')
            self.rescaler['None'] = Rescaler(value_max=None, value_min=None, remove_minus_flag=self.rm, rescale_func='norm')
        # ------------------------------------------------------- get rescaler ------------------------------------------------------- #
        # ------------------------------------------------------- get im masks ------------------------------------------------------- #
        # if self.im doesn't exist
        if len(self.im.keys())==0 and (self.fs!='all'):
            if not isinstance(select_category, list):
                    select_category = [select_category]
            for tc in select_category:
                im_path = self.im_path[str(tc)]
                if not os.path.isfile(im_path):
                    print('only for importance matrix')
                    self._analyzer_main()
            for tc in select_category:
                self.im[str(tc)], _, _ = self._get_importances(im_path, tc)
        else:
            self.im['uniform'] = None
            self.im['None'] = None
        # ------------------------------------------------------- get im masks ------------------------------------------------------- #

        # ------------------------------------------------------- call artists ------------------------------------------------------- #
        assert cam_type in ['1D', '2D', '3D', None]
        self.artist = Artists(cam_dir=self.cam_dir, cam_type=cam_type, groups=self.groups, backup=True)
        # ------------------------------------------------------- call artists ------------------------------------------------------- #
        print(f'------------------------------ initialized ------------------------------')


    def _get_importances(self, im_path, tc):
        if os.path.exists(im_path) and self.fs != 'all':
            print('loading importance matrix with mode:{}'.format(self.fs))
            im = im_reader(im_path, self.fs)
            # im [num_classes, num_features]
            im = target_cam_selection(im, mode=self.fs, extra=self.fsr)
            # im [num_classes, num_features]
        else:
            im = None
        # step 2. max-min normalization - or not
        data_max_value, data_min_value = maxmin_reader(im_path, tc)
        data_max_value, data_min_value = data_max_value, data_min_value
        return im, data_max_value, data_min_value


    def _analyzer_main(self, confidence_weight_flag:bool=False):
        if isinstance(self.select_category, list):
            for tc in self.select_category:
                im_path = self.im_path[str(tc)]
                print('im_path: {}'.format(im_path))
                if not os.path.isfile(im_path):  # only when the file dosen't exist -- because some loop would be repeated in experiments
                    print(f'--------- creating IMs for target {tc} ---------')
                    im_overall, im_target, im_diff, cam_grad_max_matrix, cam_grad_min_matrix \
                        = self._cam_stats_step(target_category=tc,
                                        confidence_weight_flag=confidence_weight_flag)
                    im_save(im_overall, im_target, im_diff, cam_grad_max_matrix, cam_grad_min_matrix,
                            im_path)
        else:
            im_path = self.im_path[str(self.select_category)]
            print('im_path: {}'.format(im_path))
            if not os.path.isfile(im_path):  # only when the file dosen't exist -- because some loop would be repeated in experiments
                if self.select_category == None or self.select_category=='GT':
                    print(f'--------- creating IMs for all categories ---------')
                else:
                    print(f'--------- creating IMs for target {self.select_category} ---------')
                im_overall, im_target, im_diff, cam_grad_max_matrix, cam_grad_min_matrix \
                    = self._cam_stats_step(target_category=self.select_category,
                                        confidence_weight_flag=confidence_weight_flag)
                im_save(im_overall, im_target, im_diff, cam_grad_max_matrix, cam_grad_min_matrix,
                        im_path)


    def _cam_stats_step(self, target_category:Union[None, int, str, list]=1, # the target category for cam
                        confidence_weight_flag:bool=False,  # if prefer to weight the importance with confidence
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        in_fold_counter = 0
        in_fold_target_counter = np.zeros([self.num_classes], dtype=np.int16)
        im_overall = None  # 400 = number of channels per group
        im_target = [0] * self.num_classes
        im_diff = [0] *self.num_classes
        # cammax & cammin for overall max-min normalization
        cam_grad_max_matrix = []
        cam_grad_min_matrix = []
        
        model = self.model.to(device=device)
        model.eval()
        
        dataset = tqdm(self.dataset) if platform.system().lower()=='windows' else self.dataset
        for x,y in dataset:
            x = x.to(dtype=torch.float32).to(device)
            y = y.to(dtype=torch.float32).to(device)
            
            grayscale_cam, predict_category, confidence, cam_grad_max_value, cam_grad_min_value\
                 = self.camoperator(input_tensor=x, target_category=target_category, gt=None, ifaoperation=True)
            # cam_single_max_value - [batch, 1*value]已经展平 --[batch]
            cam_grad_max_matrix.extend(cam_grad_max_value)
            cam_grad_min_matrix.extend(cam_grad_min_value)

            # proved: grayscale_cam - 1* [16, 512]
            if type(grayscale_cam)==list:
                grayscale_cam = grayscale_cam[0]  # [1, all_channel] remove the target layers
            # grayscale_cam - [16, 512]
            for i in range(grayscale_cam.shape[0]): # [all_channel]
                single_grayscale_cam, single_predict_category, single_confidence = grayscale_cam[i], predict_category[i], confidence[i]
                if confidence_weight_flag:
                    single_grayscale_cam = single_grayscale_cam * single_confidence
                    single_max_reviser = single_confidence
                else:
                    single_max_reviser = 1

                if self.ram:
                    y_d = y.data.cpu().numpy()[i][target_category]
                else:
                    y_d = y.data.cpu().numpy()[i]  # 只叠加正确分类的部分
                if single_predict_category == y_d:
                    # 添加总体IM
                    if im_overall is None:
                        im_overall = single_grayscale_cam
                    else:
                        im_overall = im_overall + single_grayscale_cam
                    in_fold_counter += 1
                    # 添加对应类的IM
                    if not isinstance(im_target[single_predict_category], type(np.array)):
                        im_target[single_predict_category] = np.asarray(single_grayscale_cam)
                    else:  # when list is empty
                        im_target[single_predict_category] = im_target[single_predict_category] + single_grayscale_cam
                    in_fold_target_counter[single_predict_category] += single_max_reviser
                   
        # im_target - [num_classes, num_features]
        im_overall = np.asarray(im_overall, dtype=np.float32)
        im_target = np.asarray(im_target, dtype=np.float32)
        im_overall = im_overall / in_fold_counter
        im_target = im_target / in_fold_target_counter[:, None]
        # TODO figure out how it works for None input
        for i in range(self.num_classes):
            im_diff[i] = im_target[i, :] - im_overall
        im_diff = np.asarray(im_diff, dtype=np.float32)
        # im_overall [num_out_channel]
        # im_target/im_diff [num_classes, num_out_channel]
        # 此处im不分group因为不同group的feature在heatmap上就应该不同，在重要性上的差异也应该保留
        # 而max min不分group或者类别因为需要全局统一尺度，无论group或者batch或者category

        # calculate the percentiles instead of max and min
        cam_grad_max_matrix = np.array(cam_grad_max_matrix)
        cam_grad_min_matrix = np.array(cam_grad_min_matrix)
        cam_grad_max_matrix, cam_grad_min_matrix = stat_calculator(cam_grad_max_matrix, cam_grad_min_matrix)
        return im_overall, im_target, im_diff, cam_grad_max_matrix, cam_grad_min_matrix
    

    def creator_main(self, cr_dataset:Union[None, DataLoader],
                    creator_target_category:Union[None, str, int, list]='Default',
                    # if wanted to target a category while the analyzer using None
                    eval_act:Union[bool, str]=False,
                    cam_save:bool=True, 
                    cluster:Union[None, list[int], np.array]=None,  # 同一个cluster的放到一个nii里面
                    use_origin:bool=True,  # for overlay/or not
                    max_iter:Union[None, int]=None
                    ):
        if creator_target_category=='Default':
            creator_target_category = self.select_category
            print('use default and something goes wrong with previous assertion')
            # the load im for cam creator
            # this could be None, int, list, when self.target_category == None
            # None: get the prediction-related CAMs, while the target_category for IM creation is None
            # int&list: get the CAM for certain category, while the target_category for IM creation is None
        creator_tc = creator_target_category if self.select_category==None else self.select_category
        if not isinstance(creator_tc, list):
            creator_tc = [creator_tc]
        if cluster:
            if np.sum(np.asarray(cluster))!=len(np.asarray(creator_tc)):
                raise AttributeError(f'the cluster number {np.sum(np.asarray(cluster))} given \
                                     doesnt match that of selected outputs {len(np.asarray(creator_tc))}')

        # create evaluation metric
        if eval_act is not False:
            ea  = EvalAgent(save_path=self.record_dir, eval_act=eval_act, creator_tc=creator_tc, 
                            num_classes=self.num_classes, groups=self.groups)
            logit_flag = True if ('logit' in eval_act) else False
        else:
            logit_flag = False
        # self.im_path = {}  'str(tc)'=xxx, 'uniform'=xxx
        # self.cam_dir = {}
        # self.record_dir = {}
        # self.value_max = {}
        # self.value_min = {}
        # self.im = {}

        # for cam calculation
        device = self.device
        model = self.model.to(device=device)
        model.eval()

        # -------------- start cam calculation -------------- #
        dataset = cr_dataset if cr_dataset else self.dataset
        dataset = tqdm(dataset) if platform.system().lower()=='windows' else dataset
        counter = 0
        for x,y in dataset:
            if max_iter and counter>=max_iter:
                break
            origin = x.data.numpy()  # for image generation
            # 原图的可能的维度有：batch, channel(RGB, slice), groups, slice(D), L, W
            x = x.to(dtype=torch.float32).to(device)
            y = y.to(device)

            tc_cam = []  # 仅用于需要合并不同类cam的情况下使用
            tc_pred_category = []
            tc_score = []
            tc_truth = []

            for tc in creator_tc:
                if self.rescale=='multi':
                    rescaler = self.rescaler['uniform']
                else:
                    rescaler = self.rescaler[str(tc)] if str(tc) in self.rescaler.keys() else self.rescaler['uniform']
                im = self.im[str(tc)] if str(tc) in self.im.keys() else self.im['uniform']
                grayscale_cam, predict_category, pred_score, nega_score\
                    = self.camoperator(input_tensor=x, target_category=tc, gt=y, ifaoperation=False, 
                                        im=im, out_logit=logit_flag, rescaler=rescaler)
            
                # TODO check the size of these items, avoid differences caused by ram
                # predict_category = predict_category[0] if isinstance(predict_category[0], list) else predict_category
                # pred_score = pred_score[0] if isinstance(pred_score[0], list) else pred_score
                # nega_score = nega_score[0] if isinstance(nega_score[0], list) else nega_score

                # theory: grayscale_cam -- batch * (target_layer_aggregated)_array[groups, (depth), length, width]
                # proved: grayscale_cam -- 16 * [1(groups), 256, 256] - batch * [1(groups), 256, 256]
                # therefore: grayscale_cam -- [batch, groups, (D), L, W] numpy array
                # predict_category -- [batch, 1(label)] or [batch, list[1[regression target]]]
                # pred_score -- [batch, 1(logit for one class)] or [batch, 1(logit for one output)]

                # --------------------------------------  cam evaluate  -------------------------------------- #
                if eval_act in ['corr', 'corr_logit']:
                    ea.eval(tc, rescaler.unscale(grayscale_cam), predict_category, pred_score)
                elif eval_act in ['basic', 'logit']:
                    ea.blockeval(tc, grayscale_cam, predict_category, pred_score, nega_score, 
                                      x, y, model, device)
                # the numbers are accumulated in the attributes of self.ea
                # --------------------------------------  cam evaluate  -------------------------------------- #
                    
                if cam_save:  # sometimes only want the evaluation
                    tc_cam.append(grayscale_cam)  # tc_cam: tc_len* batch* (target_layer_aggregated)_array[groups, (depth), length, width]
                    # tc_cam: [5(tc) * [16 * [1(groups), 256, 256]]] / [5(tc) * [16 * [1(groups), depth, 256, 256]]]
                    tc_pred_category.append(predict_category)
                    # tc_pred_category: [5(tc), batch, 1(argmaxed)]
                    tc_score.append(pred_score)
                    # tc_score: [5(tc), batch, 1(max logit)]
                    tc_truth.append(y.data.cpu().numpy())

            # ---------------------------------------  cam create  --------------------------------------- #
            if cam_save:  # sometimes only want the evaluation
                tc_cam = np.asarray(tc_cam)  # [tc, batch, groups, (D), L, W]
                tc_pred_category = np.asarray(tc_pred_category)
                tc_score = np.asarray(tc_score)
                tc_truth = np.asarray(tc_truth)
                if len(tc_cam.shape)==6:
                    tc_cam = np.transpose(np.asarray(tc_cam), (1, 2, 0, 3, 4, 5))   # [batch, groups, tc, (D), L, W]
                elif len(tc_cam.shape)==5:
                    tc_cam = np.transpose(np.asarray(tc_cam), (1, 2, 0, 3, 4)) # [batch, groups, tc, L, W]
                elif len(tc_cam.shape)==4:
                    tc_cam = np.transpose(np.asarray(tc_cam), (1, 2, 0, 3))   # [batch, groups, tc, W]
                # 如果集成就合并一下，如果不就挨个生成CAM
                if cluster and (max(cluster)>1):  # width-prior, merge cams required
                    camshape = tc_cam.shape
                    camshape[2] = len(cluster)
                    clustercam = np.zeros(camshape)
                    clusterpredca = np.zeros((cluster))
                    clustersc = np.zeros((cluster))
                    clustertr = np.zeros((cluster))
                    cluster_counter = 0 
                    for i in range(len(cluster)):
                        clustercam[:, :, i] = np.sum(tc_cam[:, :, cluster_counter:cluster_counter+cluster[i]], axis=2)
                        clusterpredca[i] = np.sum(tc_pred_category[cluster_counter:cluster_counter+cluster[i]])
                        clustersc[i] = np.sum(tc_score[cluster_counter:cluster_counter+cluster[i]])
                        clustertr[i] = np.sum(tc_truth[cluster_counter:cluster_counter+cluster[i]])
                        cluster_counter+=cluster[i]
                    tc_cam = clustercam  # [batch, groups/channels, cluster, (D), L, W]
                    tc_pred_category = clusterpredca  # [batch]
                    tc_score = clustersc
                    tc_truth = clustertr
                
                if len(tc_pred_category.shape)>1:  # squeeze only when have multiple batch/cluster
                    tc_pred_category = np.squeeze(tc_pred_category, axis=0)
                    tc_score = np.squeeze(tc_score, axis=0)
                    tc_truth = np.squeeze(tc_truth, axis=0)
                # [batch, groups/channels, cluster/tc, (D), L, W]
                # 只有两种情况：存2D和存3D，但是根据输入的维度在合并时有出入：
                    # 所有1D数据[batch, groups/channels, cluster/tc, W] 转2D
                    # 输入2D存2D，可以直接生成，对batch和cluster循环即可
                    # 输入2D存3D，存不了
                    # 输入3D存3D，生成两个nii文件
                    # 输入3D存2D，需要分别扫描件
                # origin [batch, channel(groups), (D), L, W]  # origin 需要区分一下channel和group--但是通常有group的没有channel
                # tc_cam [batch, groups, cluster/tc, (D), L, W]
                B, _, C = tc_cam.shape[:3]
                for b in range(B):
                    counter+=1  # different samples
                    for c in range(C):
                        pr_str = str(tc_pred_category[b])[:4] if len(str(tc_pred_category[b]))>4 else str(tc_pred_category[b])
                        tr_str = str(tc_truth[b])[:4] if len(str(tc_truth[b]))>4 else str(tc_truth[b])
                        ps_str = str(pred_score[b])[:4] if len(str(pred_score[b]))>4 else str(pred_score[b])
                        name_str = os.path.join(self.cam_dir[str(creator_tc[c])], f'{counter}_pr{pr_str}_tr{tr_str}_cf{ps_str}')
                        self.artist.img_create(cam=tc_cam[b, :, c, ], origin=origin[b, :], 
                                               # [batch, groups, tc, (D), L, W] -> [groups, (D), L, W], 
                                               # [batch, channel(groups), (D), L, W] -> [channel(groups), (D), L, W]
                                               name_str=name_str,
                                               use_origin=use_origin)
                        # input: [channel(groups*n), (D), L, W] & [groups, (D), L, W]
            # ---------------------------------------  cam create  --------------------------------------- #

        # --------------------------------------  cam evaluate summary  -------------------------------------- #
        if eval_act is not False:
            for tc in creator_tc:  
                ea.evalsummary(tc)

            
    def indiv_return(self, x:torch.Tensor,  # make sure the input is a 4-dimension tensor
                    creator_target_category:Union[None, str, int, list]='Default',
                    cluster:Union[None, list[int], np.array]=None,
                    pred_flag:bool=False) -> np.array:  # [batch, (slice), L, W]
        # use origin -- False
        # use eval -- False
        if creator_target_category=='Default':
            creator_target_category = self.select_category
            print('use default and something goes wrong with previous assertion')
            # the load im for cam creator
            # this could be None, int, list, when self.target_category == None
            # None: get the prediction-related CAMs, while the target_category for IM creation is None
            # int&list: get the CAM for certain category, while the target_category for IM creation is None
        creator_tc = creator_target_category if self.select_category==None else self.select_category
        if not isinstance(creator_tc, list):
            creator_tc = [creator_tc]
        if cluster:
            if np.sum(np.asarray(cluster))!=len(np.asarray(creator_tc)):
                raise AttributeError(f'the cluster number {np.sum(np.asarray(cluster))} given \
                                     doesnt match that of selected outputs {len(np.asarray(creator_tc))}')

        # for cam calculation
        device = self.device
        model = self.model.to(device=device)
        model.eval()

        # -------------- start cam calculation -------------- #
        x = x.to(dtype=torch.float32).to(device)   # [batch, groups/channels, (D), L, W]
        tc_cam = []  # 仅用于需要合并不同类cam的情况下使用
        for tc in creator_tc:
            if self.rescale=='multi':
                    rescaler = self.rescaler['uniform']
            else:
                rescaler = self.rescaler[str(tc)] if str(tc) in self.rescaler.keys() else self.rescaler['uniform']
            im = self.im[str(tc)] if str(tc) in self.im.keys() else self.im['uniform']
            grayscale_cam, pred_category, _, _\
                = self.camoperator(input_tensor=x, target_category=tc, gt=None, ifaoperation=False, 
                                    im=im, out_logit=False, rescaler=rescaler)
            tc_cam.append(grayscale_cam)  # tc_cam: tc_len* batch* (target_layer_aggregated)_array[groups, (depth), length, width]
                # tc_cam: [5(tc) * [16 * [1(groups), 256, 256]]] / [5(tc) * [16 * [1(groups), depth, 256, 256]]]
        # [tc, batch, groups, (D), L, W]
        tc_cam = np.asarray(tc_cam)
        if len(tc_cam.shape)==6:
            tc_cam = np.transpose(tc_cam, (1, 2, 0, 3, 4, 5))   # [batch, groups, tc, (D), L, W]
        elif len(tc_cam.shape)==5:
            tc_cam = np.transpose(tc_cam, (1, 2, 0, 3, 4)) # [batch, groups, tc, L, W]
        elif len(tc_cam.shape)==4:
            tc_cam = np.transpose(tc_cam, (1, 2, 0, 3))   # [batch, groups, tc, W]
        # [batch, groups, tc, (D), L, W]
        if cluster and (max(cluster)>1):  # width-prior, merge cams required
            camshape = tc_cam.shape
            camshape[2] = len(cluster)
            clustercam = np.zeros(camshape)
            cluster_counter = 0 
            for i in range(len(cluster)):
                clustercam[:, :, i] = np.sum(tc_cam[:, :, cluster_counter:cluster_counter+cluster[i]], axis=2)
                cluster_counter+=cluster[i]
            tc_cam = clustercam  # [batch, groups/channels, cluster, (D), L, W]
        # [batch, groups, cluster, (D), L, W]
        return tc_cam, pred_category if pred_flag else tc_cam  # [batch, groups, cluster/tc, (D), L, W]
        # [1, 1, cluster/tc, (D), L, W], [batch, 1(label)]
            
