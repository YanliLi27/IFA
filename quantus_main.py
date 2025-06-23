from otherutil.quantus_func import WorkSpace


if __name__=='main':
    tasks:list = ['catsdogs', 'imagenet', 'mnist', 'luna', 'rsna', 'siim', 'us', 'esmira']
    apply_norm:list = [True, False]
    methods:list = ['gradcam', 'fullcam', 'gradcampp', 'xgradcam']
    for task in tasks:
        for method in methods:
            for apply in apply_norm:
                ws = WorkSpace(task=task, method=method, apply_norm=apply)
                ws.run_quantus()