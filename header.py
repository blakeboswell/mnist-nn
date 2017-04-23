import os
import numpy as np

DATAROOT = '../tensorflow-pg/data'


def data_path(*tokens):
    ''' join tokens with data root
    '''
    return os.path.join(DATAROOT, *tokens)


def train_message(epoch_i, metric):
    ''' print general message during training
    '''
    print('epoch {}:  score: {}'.format(epoch_i, metric))

def max_message(metrics):
    ''' print max and argmax of np array of metrics
    '''
    n = np.max(metrics)
    i = np.argmax(metrics)
    print('max {} at epoch {}'.format(n, i))
