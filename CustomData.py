import caffe
import numpy as np
import pdb 

class CustomData(caffe.Layer):
    """ LOAD CUSTOM DATA FROM PYTHON BECAUSE MEMORYDATALAYER DOESN'T WORK"""
       
    def setup(self, bottom, top):
        vals = [int(x) for x in self.param_str.split(',')]
        self.MY_TOP_SHAPE = tuple(vals)

    def reshape(self, bottom, top):
        # allocate memory for the top
        top[0].reshape(*self.MY_TOP_SHAPE)

    def forward(self, bottom, top):
        # data is set by driver
        pass

    def backward(self, top, propagate_down, bottom):
        # this is a data layer - i.e. no backward
        pass