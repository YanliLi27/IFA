# Integrated feature analysis (Unfinished version)

## The code for "Integrated feature analysis for deep learning interpretation and class_activation_maps"

## The CAMs for signal processing were also included, refering to "A lightweight model for physiological signals-based sleep staging with multiclass CAM for model explainability", Y. Yang, Y. Li.

> Thanks to https://github.com/frgfm/torch-cam and https://github.com/jacobgil/pytorch-grad-cam for their functions.

> Currently, default test support for MNIST, ILSVRC2012, Cats&Dogs, and other four public medical datasets. ESMIRA (private data) is not supported as it includes the information of patients.


## Use:
### 0. **For simple use** ###
#### Step1: Copy the folder: cam_components into your project.
#### Step2: from cam_components.camagent import CAMAgent
#### Step3: Give the following information to the class, most of them are default.
> Agent = CAMAgent(model, target_layer, dataset, groups, ram=False, cam_method=method, name_str=f'{task}_{fold_order}',
>                         batch_size=batch_size, select_category=target_category, rescale='norm',  remove_minus_flag=True, scale_ratio=1,
>                         feature_selection='all', feature_selection_ratio=1.0, cam_type='2D')
#### Step4: Input your data and get the rescaled CAMs.
> indiv_cam = Agent.indiv_return(x, target_category, None)
#### Step5: For feature analysis, you can find the matrices in './output/*namestr*/im'


### 1. main.py provides some examples of runners, with some predefined tasks and datasets that were presented in the manuscript.
> Find them in the ./runner.
> In main.py, examples were given for generating CAMs of MNIST, ILSVRC2012, Cats&Dogs and other four medical image tasks with the default paths.


### 2. **Add more CAM methods, please see the /cam_components/methods/*cam.py**



### 3. **Change the functions for importance matrices and evaluation, see /cam_components/metric/*.py**



### 4. For the output, you can create a dir named output for collection, the default is ./output/*namestr*/im&cam&figs.

