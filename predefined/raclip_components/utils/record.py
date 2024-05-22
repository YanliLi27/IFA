import os

def record_save(metric1, metric2, epoch:int, save_path:str='./models/figs/record.txt', highlight:bool=False) ->None:
    if not os.path.isfile(save_path):
        with open(save_path, 'w') as F:
            F.write('fold order:{}\n'.format(epoch))
    if highlight:
        with open(save_path, 'a') as F:
            F.write('\n')
            F.write('-------------------------------- final ------------------------------')
    with open(save_path, 'a') as F:
        F.write('\n')
        F.write(f'epoch: {epoch}')
        F.write('\n')
        F.write(f'mse: {str(metric1)[:8]}')
        F.write('\n')
        F.write('divided mse')
        F.write('\n')
        F.write(str(metric2))
        F.write('\n')
        F.write('---------------- Next ------------------')
    print(f'record saved with best metric: {metric1}')


def auc_save(metric1, epoch, save_path:str='./models/figs/record.txt') ->None:
    if not os.path.isfile(save_path):
        with open(save_path, 'w') as F:
            F.write('fold order:{}\n'.format(epoch))
    with open(save_path, 'a') as F:
        F.write('\n')
        F.write(f'epoch: {epoch}')
        F.write('\n')
        F.write(f'auc: {str(metric1)}')
        F.write('\n')
        F.write('---------------- Next ------------------')
        F.write('\n')
    print(f'record saved with best auc: {metric1}')



def corr_save(corr, p_value, mode='corr', save_path:str='./models/figs/record.txt', highlight:bool=False) ->None:
    if not os.path.isfile(save_path):
        with open(save_path, 'w') as F:
            F.write('fold order 0')
    if highlight:
        with open(save_path, 'a') as F:
            F.write('\n')
            F.write('-------------------------------- final ------------------------------')
    if mode=='corr':
        with open(save_path, 'a') as F:
            F.write('\n')
            F.write(f'corr: {str(corr)[:8]}')
            F.write('\n')
            F.write(f'p_value: {str(p_value)}')
    elif mode=='mse':
        with open(save_path, 'a') as F:
            F.write('\n')
            F.write(f'mse and pred corr: {str(corr)[:8]}')
            F.write('\n')
            F.write(f'mse and pred corr p_value: {str(p_value)}')
    elif mode=='acc':
        with open(save_path, 'a') as F:
            F.write('\n')
            F.write(f'acc: {str(corr)}')
    elif mode=='corr_regu':
        with open(save_path, 'a') as F:
            F.write('\n')
            F.write(f'regularized corr: {str(corr)[:8]}')
            F.write('\n')
            F.write(f'regularized p_value: {str(p_value)}')
    elif mode=='icc':
        with open(save_path, 'a') as F:
            F.write('\n')
            F.write(f'icc: {str(corr)[:8]}')
            F.write('\n')
            F.write(f'regu_icc: {str(p_value)[:8]}')