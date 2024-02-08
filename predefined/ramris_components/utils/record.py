import os

def record_save(metric1, metric2, epoch:int, save_path:str='./models/figs/record.txt') ->None:
    if not os.path.isfile(save_path):
        with open(save_path, 'w') as F:
            F.write('fold order:{}\n'.format(epoch))
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
    print(f'record saved with best mse: {metric1}')


def corr_save(corr, p_value, mode='corr', save_path:str='./models/figs/record.txt') ->None:
    if not os.path.isfile(save_path):
        with open(save_path, 'w') as F:
            F.write('fold order 0')
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