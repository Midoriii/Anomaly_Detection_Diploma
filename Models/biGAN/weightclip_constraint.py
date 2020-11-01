'''
Util class, is helpful when using Wasserstein loss.
'''
from keras.constraints import Constraint
from keras import backend as K


class WeightClip(Constraint):
    '''
    Custom kernel constraint which clips the weights into range [-c,c].
    '''
    def __init__(self, c=0.01):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}
