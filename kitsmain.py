from runner.kits_runner import kits_runner

if __name__ == '__main__':
    model_zoo = ['csv', 'modelclass']
    for model in model_zoo:
        kits_runner(10000, False, model, r'D:\ESMIRAcode\ACAM\model\logs\bestmodelkits.model')