import matplotlib.pyplot as plt
import os


def roc_printer(fpr, tpr, auc_value, fold_order, epoch_order, output_path):
    lw = 2
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc_value)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characeristic example')
    plt.legend(loc='lower right')
    fig_path = os.path.join(output_path, 'output/fold{}_epoch{}_roc_contra.png'.format(fold_order, epoch_order))
    plt.savefig(fig_path)