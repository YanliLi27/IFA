from runner.kits_runner import kits_runner

if __name__ == '__main__':
    model_zoo = ['csv'] #'modelclass', 'simplecsv'] # , 
    weight_zoo = {'csv':r'D:\ESMIRAcode\ACAM\model\logs\imagein10000_csv\bestmodelkits.model',
                  'simplecsv':r'D:\ESMIRAcode\ACAM\model\logs\imagein10000_simplecsv\bestmodelkits.model',
                  'modelclass':r'D:\ESMIRAcode\ACAM\model\logs\bestmodelkits.model'}
    for model in model_zoo:
        kits_runner(10000, False, weight_zoo[model],  model)