from otherutil.quantus_func import WorkSpace


if __name__=='__main__':
    tasks:list = ['ddsm'] # 'mnist', 'imagenet', 'mnist', 'siim', 'luna', 'catsdogs', 'rsna', 'us', 'esmira']#,  ']
    apply_norm:list = [False, True]
    methods:list = ['gradcam', 'fullcam', 'gradcampp', 'xgradcam']
    for task in tasks:
        for method in methods:
            for apply in apply_norm:
                if task=='imagenet':
                    num_classes = 1000
                    subset_size = 32
                    maxiter = 1000
                elif task=='mnist':
                    num_classes = 10
                    subset_size = 2
                    maxiter = None
                else:
                    num_classes = 2
                    subset_size = 32
                    maxiter = 500
                ws = WorkSpace(task=task, method=method, apply_norm=apply, num_classes=num_classes, subset_size=subset_size)
                ws.run_quantus(max_iter=maxiter)