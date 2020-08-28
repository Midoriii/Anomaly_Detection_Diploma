'''
Custom losses for Siamese nets, namely Triplet Loss and Contrastive Loss
'''

from keras import backend as K



def contrastive_loss(y_true, y_pred):
    '''
    Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    # Custom margin which should be attained between disimilar pairs.
    # In our case, since we sigmoid the euclidean distance between feature
    # spaces, the output is squashed between [0,1], so the maximum margin that
    # makes sense is 1.
    margin = 1
    # Loss itself
    return K.mean((1 - y_true) * K.square(y_pred)
                  + y_true * K.square(K.maximum(margin - y_pred, 0)))
