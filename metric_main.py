from runner.predefined_runner import naturalimage_runner, catsdog3d_runner, esmira_runner, medical_runner
from runner.reg_runner import ramris_pred_runner
from runner.indiv_runner import indiv_runner



if __name__ == '__main__':
    eval_act = 'insdel'
    # for natural images test 
    task_zoo = [ 'Imagenet'] # 'MNIST', 'CatsDogs']   #
    model_zoo = {'CatsDogs':'vgg', 'Imagenet':'vgg', 'MNIST':'scratch_mnist'}
    tc_zoo = {'CatsDogs':None, 'Imagenet':None, 'MNIST':None}

    for task in task_zoo:
        model = model_zoo[task]
        tc = tc_zoo[task]
        if task == 'Imagenet':
            cam_method_zoo = ['gradcam', 'fullcam', 'gradcampp', 'xgradcam']
        else:
            cam_method_zoo = ['gradcam', 'fullcam', 'gradcampp', 'xgradcam']
        naturalimage_runner(target_category=None, model_flag=model, task=task, dataset_split='val',
                            max_iter=5000, randomization=False, random_severity=0,
                            mm_rm_setting=[['norm', True], [False, True]],
                            eval_flag=eval_act, tan_flag=False, cam_method=cam_method_zoo,
                            cam_save=True)

    # task_zoo = ['luna', 'rsna', 'siim', 'us', 'ddsm' ]
    # for task in task_zoo:
    #     medical_runner(target_category=None, task=task, dataset_split='val', 
    #                    mm_rm_setting=[['norm', True], [False, True]],
    #                    cam_save=False, eval_flag=eval_act)
    
    # esmira_runner(target_category=None, data_dir='D:\\ESMIRA\\ESMIRA_common',
    #               target_catename=['CSA'], target_site=['Wrist'], target_dirc=['TRA', 'COR'], 
    #               mm_rm_setting=[['norm', True], [False, True]],
    #               cam_save=True, eval_flag=eval_act)

