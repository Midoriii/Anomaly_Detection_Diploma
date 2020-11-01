'''
Custom losses for Siamese nets, namely Triplet Loss and Contrastive Loss.
Also contains Hinge Loss and Wasserstein Loss for biGAN.
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


def triplet_loss(y_true, y_pred, alpha=10.0):
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
    batch_length = y_pred.shape.as_list()[-1]
    # Unpack the y_pred
    anchor = y_pred[:, 0:int(batch_length * 1/3)]
    pos = y_pred[:, int(batch_length * 1/3):int(batch_length * 2/3)]
    neg = y_pred[:, int(batch_length * 2/3):int(batch_length)]
    # Calculate Square Euclidean distances
    pos_distance = K.sum(K.square(anchor - pos), axis=1)
    neg_distance = K.sum(K.square(anchor - neg), axis=1)
    # Return loss
    return K.mean(K.maximum(0.0, pos_distance - neg_distance + alpha))


def hinge_loss(y_true, y_pred):
    '''
    A loss used with Classifiers. It should be noted that this loss expects y
    to be a raw number, not sigmoided one.
    Viz: https://en.wikipedia.org/wiki/Hinge_loss

    Arguments:
        y_true: True labels of samples - A list of either -1 (Fake) or 1 (Real).
        y_pred: A list of raw, un-sigmoided outputs of the classifier.
    '''
    return K.mean(K.maximum(0.0, 1.0 - (y_true * y_pred)))


def wasserstein_loss(y_true, y_pred):
    '''
    https://machinelearningmastery.com/how-to-implement-wasserstein-loss-for-generative-adversarial-networks/

    Arguments:
        y_true: True labels of samples - A list of either -1 (Fake) or 1 (Real).
        y_pred: A list of raw, un-sigmoided outputs of the classifier.
    '''
    return K.mean(y_true * y_pred)
