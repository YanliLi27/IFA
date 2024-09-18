# Integrated feature analysis (Unfinished version)

<p > A pytorch package to calculate the calibrated CAMs and analyze datasets based on trained models  </p>
<p > Input support: 2D, 3D images and signals, Output: Single/Multiple/Clustered output </p>



## Use:
### 0. **For simple use** ###
#### Step1: Setup the module: cam_components into your project.
```bash
python ifasetup.py install
```
#### Step2: Import the package, set your model and dataset
```python
# Initialize your model
from cam_components.camagent import CAMAgent
model:nn.Module = 'your model'
target_layer:list = [model.feature[-1]]  # the layer/layers for obtaining heatmaps
dataset = 'your dataset'  # Dataset, not Dataloader
```
#### Step3: Give the following information to the class, most of them are default.
```python
Agent = CAMAgent(model,   # your model
                 target_layer,  # the layer/layers for obtaining heatmaps
                 dataset,  # Dataset, not Dataloader
                 groups=1,  # group conv in your model
                 ram=False,  # for regression tasks, regression activation mapping
                 cam_method=method,  # ['gradcam', , 'fullcam', 'gradcampp', 'xgradcam', ... ] more in './cam_components/methods/*cam.py'
                 name_str=f'{your_task}',  # output name: './output/*namestr*/im'
                 batch_size=batch_size,  # for acceleration
                 select_category=target_category,  # default to be 0, the target category in your task
                 rescale='norm',  # ['norm', 'tanh']  different method for calibration and rescaling
                 remove_minus_flag=True,  # If only keep the values above 0 in orginal weighted heatmaps
                 scale_ratio=1,  # for better visualization
                 feature_selection='all',  # ['reverse_diff_top', 'max', 'top', 'diff_top', 'freq', 'index', 'all'] if select features for visualization
                 feature_selection_ratio=1.0,  # The ratio of selected features/all features
                 cam_type='2D')  # Output dimension.
```
#### Step4: Input your data and get the rescaled CAMs.
```python
indiv_cam = Agent.indiv_return(x, target_category, None)
```
#### Step5: Input your data and get the rescaled CAMs at a dataset scale with analysis.
```python
Agent.creator_main(dataset,   # your dataset, optional
                   [categories],    # select the categories
                   eval_act='corr',    # type of evaluation: 'corr':correlation, 'basic':mask-based evaluation, 'corr_logit':correlation base on logits
                   cam_save=True,  # If save the heatmaps in './output/*namestr*/cam'
                   cluster=None,   # If merge the results of multiple outputs
                   use_origin=False,   # If overlay the original images and the heatmaps
                   max_iter=None)   # early stop
```

#### Step6: For feature analysis, you can find the matrices in './output/*namestr*/im'


### 1. main.py provides some examples of runners, with some predefined tasks and datasets that were presented in the manuscript.
> Find them in the ./runner.
> In main.py, examples were given for generating CAMs of MNIST, ILSVRC2012, Cats&Dogs and other four medical image tasks with the default paths.

### 2. **Add more CAM methods, please see the './cam_components/methods/*cam.py'**



> for cite this repositry, please cite: Integrated feature analysis for calibrating class activation maps and beyond, arxiv.org/abs/2407.01142

### The CAMs for signal processing were also included, 
> refering to "A lightweight model for physiological signals-based sleep staging with multiclass CAM for model explainability", Y. Yang, Y. Li.

### The CAMs for 3D multi-output, multi-cluster, multi-layer images were also included,
> refering to "Automatic segmentation-free inflammation scoring in rheumatoid arthritis from MRI using deep learning", Y. Li, et. al.

### Thanks
> Thanks to https://github.com/frgfm/torch-cam and https://github.com/jacobgil/pytorch-grad-cam for their functions.

> Currently, default test support for MNIST, ILSVRC2012, Cats&Dogs, and other four public medical datasets. ESMIRA (private data) is not supported as it includes the information of patients.

### 3. **Change the functions for importance matrices and evaluation, see /cam_components/metric/*.py**



### 4. For the output, you can create a dir named output for collection, the default is ./output/*namestr*/im&cam&figs.

