from runner.predefiend_runner import naturalimage_runner, catsdog3d_runner, esmira_runner, medical_runner
from runner.reg_runner import ramris_pred_runner
from runner.indiv_runner import indiv_runner
from runner.ramris3d_runner import ramris3d_pred_runner
import platform


if __name__ == '__main__':
    # for natural images test 
    # task_zoo = ['CatsDogs'] #'MNIST', 'Imagenet','CatsDogs'] 
    # model_zoo = {'CatsDogs':'vgg', 'Imagenet':'vgg', 'MNIST':'scratch_mnist'}
    # tc_zoo = {'CatsDogs':[1], 'Imagenet':[10,11,12,13], 'MNIST':[0,1,2,3,4,5,6,7,8,9]}

    # for task in task_zoo:
    #     if task!='Imagenet':
    #         tan_flag_zoo = [False]
    #     else:
    #         tan_flag_zoo = [False]
    #     for tan_flag in tan_flag_zoo:
    #         model = model_zoo[task]
    #         tc = tc_zoo[task]
    #         if task == 'Imagenet':
    #             cam_method_zoo = ['gradcam', 'fullcam', 'gradcampp', 'xgradcam']
    #         else:
    #             cam_method_zoo = ['gradcam', 'fullcam', 'gradcampp', 'xgradcam']
    #         # naturalimage_runner(target_category=None, model_flag=model, task=task, dataset_split='val',
    #         #                     max_iter=None, randomization=False, random_severity=0,
    #         #                     eval_flag='basic', tan_flag=tan_flag, cam_method=cam_method_zoo,
    #         #                     cam_save=True)
    #         for tc_s in tc:
    #             naturalimage_runner(target_category=tc_s, model_flag=model, task=task, dataset_split='val',
    #                                 max_iter=None, randomization=False, random_severity=0,
    #                                 eval_flag='corr_logit', tan_flag=tan_flag, cam_method=cam_method_zoo,
    #                                 cam_save=True)


    # catsdog3d_runner(target_category=1, task='catsdogs3d', dataset_split='val')

    # task_zoo = ['luna', 'rsna', 'siim', 'us', 'ddsm' ]
    # tc_zoo = [0, 1]
    # for task in task_zoo:
    #     for tc in tc_zoo:
    #         medical_runner(target_category=None, task=task, dataset_split='val', cam_save=False, eval_flag='basic')
    #         # medical_runner(target_category=tc, task=task, dataset_split='val', cam_save=False, eval_flag='corr_logit')

    # esmira_runner(target_category=None, data_dir='D:\\ESMIRA\\ESMIRA_common',
    #               target_catename=['CSA'], target_site=['Wrist'], target_dirc=['TRA', 'COR'], cam_save=True, eval_flag='basic')
    # tc_zoo = [0, 1]
    # for tc in tc_zoo:
    #     esmira_runner(target_category=tc, data_dir='D:\\ESMIRA\\ESMIRA_common',
    #                   target_catename=['CSA'], target_site=['Wrist'], target_dirc=['TRA', 'COR'], cam_save=True, eval_flag='corr_logit')

    # ramris_pred_runner(data_dir='', target_category=['EAC'], 
    #              target_site=['Wrist'], target_dirc=['TRA', 'COR'],
    #              target_biomarker=['SYN'],
    #              target_reader=['Reader1', 'Reader2'], task_mode='clip', phase='train',
    #              full_img=True)

    # list_of_output = [item for item in range(43)]
    # ramris_pred_runner(data_dir='D:\\ESMIRA\\ESMIRA_common', target_category=None, 
    #              target_site=['Wrist'], target_dirc=['TRA', 'COR'],
    #              target_biomarker=None,
    #              target_reader=['Reader1', 'Reader2'], task_mode='clip', phase='train',
    #              full_img=True, dimension=2,
    #              target_output=list_of_output)

    # list_of_output = [item for item in range(10)]
    # ramris_pred_runner(data_dir='D:\\ESMIRA\\ESMIRA_common', target_category=None, 
    #              target_site=['Wrist'], target_dirc=['TRA', 'COR'],
    #              target_biomarker=['TSY'],
    #              target_reader=['Reader1', 'Reader2'], task_mode='clip', phase='train',
    #              full_img=True, dimension=2,
    #              target_output=list_of_output,
    #              cluster=None, cluster_start=0)

    # list_of_output = [item for item in range(15)]
    # ramris_pred_runner(data_dir='D:\\ESMIRA\\ESMIRA_common', target_category=None, 
    #              target_site=['Wrist'], target_dirc=['TRA', 'COR'],
    #              target_biomarker=['BME'],
    #              target_reader=['Reader1', 'Reader2'], task_mode='clip', phase='train',
    #              full_img=True, dimension=2,
    #              target_output=list_of_output,
    #              cluster=None, cluster_start=0, tanh=False)


    # for cam aggregation test
    # task_zoo = ['CatsDogs'] #'MNIST', 'Imagenet','CatsDogs'] 
    # model_zoo = {'CatsDogs':'vgg'}
    # tc_zoo = {'CatsDogs':[1]}

    # for task in task_zoo:
    #     model = model_zoo[task]
    #     tc = tc_zoo[task]
    #     cam_method_zoo = ['fullcam'] #, 'gradcam', 'gradcampp', 'xgradcam']
    #     # indiv_runner(target_category=None, model_flag=model, task=task, dataset_split='val',
    #     #                     max_iter=None, randomization=False, random_severity=0,
    #     #                     eval_flag='basic', tan_flag=False, cam_method=cam_method_zoo,
    #     #                     cam_save=True)
    #     for tc_s in tc:
    #         indiv_runner(target_category=tc_s, model_flag=model, task=task, dataset_split='val',
    #                             max_iter=None, randomization=False, random_severity=0,
    #                             eval_flag='corr_logit', tan_flag=False, cam_method=cam_method_zoo,
    #                             cam_save=True)
    data_dir = 'E:\\ESMIRA_RAprediction\\Export20Jun22' if platform.system().lower()=='windows' else '/exports/lkeb-hpc/yanli/Export20Jun22'
    score_sum = True
    dim = 3
    model_csv = True
    biomakrer_zoo = [['SYN', 'TSY', 'BME']] # 'SYN', 'TSY', 'BME',  ['SYN', 'TSY', 'BME'] #['TSY']#['SYN'], ['TSY'], ['BME']] #  , 
    site_zoo = [['Wrist'], ['MCP'], ['Foot']]  #  , ['Wrist', 'MCP', 'Foot']]#,
    drc_zoo = [{'SYN':'TRA', 'TSY':'TRA', 'BME':'COR'}, {'SYN':'TRA', 'TSY':'TRA', 'BME':'COR'}, {'SYN':'COR', 'TSY':'COR', 'BME':'TRA'}]
    for n, site in enumerate(site_zoo):
        for bio in biomakrer_zoo:
            # if n<1 and isinstance(bio, str) and bio in ['SYN', 'TSY']:
            #     print('======================================================1=======================================================')
            #     continue
                ramris3d_pred_runner(data_dir=data_dir, target_category=None, target_site=site, target_dirc=['TRA', 'COR'], 
                                    target_biomarker=bio, target_reader=['Reader1', 'Reader2'],
                                    task_mode='clip', phase='train',
                                    full_img=7, dimension=3, 
                                    target_output=[0],
                                    cluster=None,
                                    tanh=False,  
                                    model_csv=model_csv, extension=0,
                                    score_sum=True, maxfold=1)