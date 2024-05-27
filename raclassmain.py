from runner import raclass_runner


if __name__ == '__main__':
    model_zoo = ['modelclass3d'] #'modelclass']#, 'csv3d', 'convsharevit', 'vit', 'mobilevit', 'mobilenet'
    raclass_runner(data_dir='D:\\ESMIRA\\CSA_resplit\\train', target_category=['CSA'], 
                target_site=['Wrist'], target_dirc=['TRA', 'COR'], phase='train',
                model_counter=model_zoo[0], attn_type='normal',
                full_img=7, maxfold=5, tanh=True)
