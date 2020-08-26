'''
Custom losses for Siamese nets, namely Triplet Loss and Contrastive Loss
'''

import numpy as np



def contrastive_loss(y_true, y_pred):
    '''
    Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    # Custom margin which should be attained between disimilar pairs
    margin = 1
    # Loss itself
    return np.mean(y_true * np.square(y_pred)
                   + (1 - y_true) * np.square(np.maximum(margin - y_pred, 0)))
