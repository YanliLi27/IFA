from runner.ramris3d_runner import ramris3d_pred_runner
from runner.synaug_runner import synaug_pred_runner
import platform


if __name__ == '__main__':
    data_dir = 'E:\\ESMIRA_RAprediction\\Export20Jun22' if platform.system().lower()=='windows' else '/exports/lkeb-hpc/yanli/Export20Jun22'
    score_sum = True
    dim = 3
    model_csv = True
    biomakrer_zoo = [['TSY'], ['SYN'], ['BME']] # 'SYN', 'TSY', 'BME',  ['SYN', 'TSY', 'BME'] #['TSY']#['SYN'], ['TSY'], ['BME']] #  , 
    site_zoo = [['Wrist'], ['MCP'], ['Foot']]  #  , ['Wrist', 'MCP', 'Foot']]#,
    # drc_zoo = [{'SYN':'TRA', 'TSY':'TRA', 'BME':'COR'}, {'SYN':'TRA', 'TSY':'TRA', 'BME':'COR'}, {'SYN':'COR', 'TSY':'COR', 'BME':'TRA'}]
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
            
    # synaug -- for denis' augmentation
    synaug_pred_runner(target_category=None, target_site=['Wrist'], target_dirc=['TRA', 'COR'],
                        target_biomarker=['TSY'],
                        full_img=7, dimension=3,
                        target_output=[0],
                        tanh=True,  
                        model_csv=True, extension=0, 
                        score_sum=False, maxfold=1)