import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Seed the random number generator.
# This ensures that the results below are reproducible.
val_path = 'D:\\Personal\\Desktop\\IM_Cats&Dogs_result\\fold0_im_test.csv'
train_path = 'D:\\Personal\\Desktop\\IM_Cats&Dogs_result\\fold0_im.csv'

val_importance_matrix = pd.read_csv(val_path)
val_im_target = val_importance_matrix['Target_Im_0']
val_im_atlas = val_importance_matrix['Atlas_Im_0']

train_importance_matrix = pd.read_csv(train_path)
train_im_target = train_importance_matrix['Target_Im_0']
train_im_atlas = train_importance_matrix['Atlas_Im_0']


f, ax = plt.subplots(1, figsize = (8,5))
plt.title('Bland altman of Cats and Dogs in Val',fontsize='large', fontweight='bold')
sm.graphics.mean_diff_plot(val_im_target, val_im_atlas, ax = ax)
plt.show()


f, ax = plt.subplots(1, figsize = (8,5))
plt.title('Bland altman of Cats and Dogs in Train',fontsize='large', fontweight='bold')
sm.graphics.mean_diff_plot(train_im_target, train_im_atlas, ax = ax)
plt.show()


f, ax = plt.subplots(1, figsize = (8,5))
plt.title('Bland altman of Dogs in Train and Val',fontsize='large', fontweight='bold')
sm.graphics.mean_diff_plot(val_im_target, train_im_target, ax = ax)
plt.show()

f, ax = plt.subplots(1, figsize = (8,5))
plt.title('Bland altman of Cats in Train and Val',fontsize='large', fontweight='bold')
sm.graphics.mean_diff_plot(val_im_atlas, train_im_atlas, ax = ax)
plt.show()