import os

DATAROOT = '../tensorflow-pg/data'


def data_path(*tokens):
    ''' join tokens with data root
    '''
    return os.path.join(DATAROOT, *tokens)


def train_message(epoch_i, metric):
    ''' print general message during training
    '''
    print('epoc {}:  score: {}'.format(epoch_i, metric))

