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


def triplet_loss(y_true, y_pred, alpha=1.0):
    '''
    A basic triplet loss function

    Arguments:
        y_true: Required by Keras, not used here.
        y_pred: A list of three numpy arrays, representing embeddings of Anchor,
        Positive and Negative Input.
        alpha: Desired margin to be learned between anchor-pos and anchor-neg distances.

    Returns:
        loss: A real number, value of loss.
    '''
    # Unpack the y_pred
    anchor = y_pred[0]
    pos = y_pred[1]
    neg = y_pred[2]
    # Calculate Square Euclidean distances
    pos_distance = K.sum(K.square(anchor - pos), axis=-1)
    neg_distance = K.sum(K.square(anchor - neg), axis=-1)
    # Return loss
    return K.maximum(0.0, pos_distance - neg_distance + alpha)
