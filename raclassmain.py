from runner.raclass_runner import raclass_pred_runner


if __name__ == '__main__':
    model_zoo = ['modelclass3d'] #'modelclass']#, 'csv3d', 'convsharevit', 'vit', 'mobilevit', 'mobilenet'
    raclass_pred_runner(data_dir='D:\\ESMIRA\\CSA_resplit\\train', target_category=['CSA'], 
                target_site=['Wrist'], target_dirc=['TRA', 'COR'], phase='train',
                model_counter=model_zoo[0], attn_type='normal',
                full_img=7, maxfold=5, tanh=True)
