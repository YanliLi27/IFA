# Feature analysis for proper intensity scaling and feature distinction in class activation maps (16/06/2025 updated)
</p>
<!--<h1 align="center"><b>Quantus</b></h1>-->
<h3 align="center"><b>A pytorch package to calculate the normalized CAMs and analyze datasets based on trained models</b></h3>
<p align="center">
  PyTorch
<p align="center">
  Input support: 2D, 3D images and signals, Output: Single/Multiple/Clustered output


## Initial idea:
<p align="center">
  <img width="350" src="![ExampleIFA](https://github.com/YanliLi27/IFA/blob/main/example.jpg)">
</p>

[Shortcut to arxiv draft](https://arxiv.org/abs/2407.01142)


## Table of contents
* [Library overview](#library-overview)
* [Installation](#installation)
* [Getting started](#getting-started)
* [Extension guidance](#extension)
* [Tutorials](#tutorials) (under construction)


## Library overview

### Purpose of this library


### Code structure (for customized modification)


### Metrics


### Datasets (evaluated in the paper)




## Installation
> **For simplest installation**
### ctrl+c/v the <b>cam_components<b> in the file to your project
> Normali installation
Download the files, and run:
```bash
python setup.py install
```

### Package requirements
```
python>=3.9.0
torch>=1.11.0
matplotlib
numpy
pandas
opencv-python==4.9.0.80  # for specific image load only
openpyxl==3.0.10  # for specific datasets only
scikit-image==0.22.0  # for resize only
scikit-learn  # for metrics
scipy==1.11.4  # for cross validation
SimpleITK==2.3.1   # for medical datasets
```

## Getting started
The following will give a short introduction to how to get started with the CAM feature analysis. The required materials needed:
> A model (model), inputs (x_batch), labels (y_batch)
> Customized CAM algorithm, if not you can use the classical algorithms embedded.

<details>
<summary><b><big>Step 1. Set up data and model</big></b></summary>
The first step is to have the data and model for calculating the normalized saliency maps, here we take MNIST classification task as an examples. (these predefined lines of code are simplified and adopted from ./runner/predefined_runner.py for easy use.)

```python
import torch
import torchvision
from torchvision import transforms
from cam_components.camagent import CAMAgent
  
# Enable GPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load a pre-trained model.
from predefined.natural_components.models.scratch_model import scratch_mnist
model = scratch_mnist(in_channel=in_channel, num_classes=num_classes)
# load example weights
if device.type == "cpu":
    model.load_state_dict(torch.load("weights/mnist_example.model", map_location=torch.device('cpu')))
else: 
    model.load_state_dict(torch.load("weights/mnist_example.model"))
target_layer:list = [model.conv3]

# Load datasets and make loaders.
dataset = torchvision.datasets.MNIST(root='./sample_data', download=True, transform=transforms.Compose([transforms.ToTensor()]))

# Load a batch of inputs and outputs to use for XAI evaluation.
x_batch, y_batch = iter(torch.utils.data.DataLoader(dataset, batch_size=24)).next()
x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()
```
</details>

<details>
<summary><b><big>Step 2. Create an agent for normalize the CAM</big></b></summary>
The second step is to create an agent to analyze the CAMs across the dataset and normalize the then generated CAMs.
> For output, you need a name of your task `your_task'.

```python
your_task:str = 'Example'  # name of the task 
agent = CAMAgent(model,   # your model
                 target_layer,  # the layer/layers for obtaining heatmaps
                 dataset,  # Dataset, not Dataloader
                 # The following attributes are typically default
                 groups=1,  # if group conv in your model
                 ram=False,  # for regression tasks, regression activation mapping
                 cam_method='fullcam',  # ['gradcam', , 'fullcam', 'gradcampp', 'xgradcam', ... ]
                 name_str=f'{your_task}',  # output name: './output/*namestr*/im'
                 batch_size=batch_size,  # for acceleration
                 select_category=select_category,  # default to be 0, the target category in your task
                 rescale='norm',  # ['norm', 'tanh']  different method for calibration and rescaling
                 remove_minus_flag=False,  # If only keep the values above 0 in orginal weighted heatmaps
                 scale_ratio=1,  # for better visualization
                 feature_selection='all',  # ['reverse_diff_top', 'max', 'top', 'diff_top', 'freq', 'index', 'all'] feature distinction
                 feature_selection_ratio=1.0,  # The ratio of selected features/all features
                 cam_type='2D')  # Output dimension.

```
</details>


<details>
<summary><b><big>App 1. Create a single normalized CAM</big></b></summary>
To create a single normalized CAM first extend it to shape of 2D [batch, 1, L, W] / 3D [batch, 1, H, L, W].

```python
x_batch, y_batch = iter(test_loader).next()
x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()
x_indiv, y_indiv = x_batch[0], y_batch[0]  # just example
x_indiv = torch.from_numpy(x_indiv[np.newaxis, :])  # .to(device) if model on device

indiv_cam = Agent.indiv_return(x_indiv, select_category)
# indiv_cam: shape of 2D [batch, 1(Group Conv), 1(category in list), L, W] / 3D [batch, 1(Group Conv), 1(category in list), H, L, W]
```
</details>



<details>
<summary><b><big>App 2. Create a series of normalized CAMs</big></b></summary>
To create a normalized CAM first extend it to shape of 2D [batch, 1, L, W] / 3D [batch, 1, H, L, W].

```python
Agent.creator_main(dataset,   # your dataset, optional - None = use dataset while initialization
                   [categories],    # select the categories
                   eval_act='corr',  # metrics calculation
                   # type of evaluation: 'corr':correlation, 'basic':mask-based evaluation, more see the paper
                   cam_save=True,  # If save the heatmaps in 'output/*namestr*/cam'
                   cluster=None,   # If merge the results of multiple outputs
                   use_origin=False,   # If overlay the original images and the heatmaps
                   max_iter=None)   # early stop steps
```
</details>

<details>
<summary><b><big>Output location</big></b></summary>
For the output, you can create a dir named output for collection, the default is './output/*namestr*/im&cam&figs'.
> Importance matrices for features: './output/*namestr*/im'
> Saved heatmaps: './output/*namestr*/cam'
> Metrics of evalution: './output/*namestr*/figs'
</details>


## Extension guidance

<details>
<summary><b><big>Examples</big></b></summary>
main.py provides some examples of runners, with some predefined tasks and datasets that were presented in the manuscript.
> Find them in the ./runner.
> In main.py, examples were given for generating CAMs of MNIST, ILSVRC2012, Cats&Dogs and other four medical image tasks with the default paths.
</details>


<details>
<summary><b><big>Customize CAM algorithm</big></b></summary>
> Add more CAM methods, please see the './cam_components/methods/*cam.py'
</details>

<details>
<summary><b><big>Customize evaluation</big></b></summary>
> Change the functions for importance matrices and evaluation, see './cam_components/metric/*.py'
</details>


### Citation
> for cite this repositry, please cite: Integrated feature analysis for deep learning interpretation and class activation maps, arxiv.org/abs/2407.01142

### The CAMs for signal processing were also included, 
> refering to "A lightweight model for physiological signals-based sleep staging with multiclass CAM for model explainability", Y. Yang, Y. Li.

### The CAMs for 3D multi-output, multi-cluster, multi-layer images were also included,
> refering to "Automatic segmentation-free inflammation scoring in rheumatoid arthritis from MRI using deep learning", Y. Li, et. al.

### Thanks
> Thanks to https://github.com/frgfm/torch-cam and https://github.com/jacobgil/pytorch-grad-cam for their functions.

> Currently, default test support for MNIST, ILSVRC2012, Cats&Dogs, and other four public medical datasets. ESMIRA (private data) is not supported as it includes the information of patients.

