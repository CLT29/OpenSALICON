import caffe
import numpy as np
import pdb 
import scipy.ndimage

class custom_interpolation_layer(caffe.Layer):
    """ INTERPOLATE THE DATA UP TO THE SIZE OF THE LARGER """
       
    def setup(self, bottom, top):
        vals = [int(x) for x in self.param_str.split(',')]
        self.INTERPOLATED_SIZE = tuple(vals)
        target_size = np.asarray(self.INTERPOLATED_SIZE, dtype=np.float32)
        current_size = np.asarray(bottom[0].data.shape, dtype=np.float32)
        self.SCALE_FACTOR = tuple(np.divide(target_size, current_size))
        self.REVERSE_SCALE_FACTOR = tuple(np.divide(current_size, target_size))
        np.seterr(divide='ignore', invalid='ignore')
        
    def reshape(self, bottom, top):
        # output is the interpolated size
        top[0].reshape(*self.INTERPOLATED_SIZE)

    def forward(self, bottom, top):
        # interpolate bottom to top
        interpolated_data = scipy.ndimage.interpolation.zoom(bottom[0].data,self.SCALE_FACTOR, np.dtype(np.float32), mode='nearest', order=1)
        top[0].data[...] = interpolated_data

    def backward(self, top, propagate_down, bottom):
        # interpolate top diff to bottom diff
        interpolated_data = scipy.ndimage.interpolation.zoom(top[0].diff,self.REVERSE_SCALE_FACTOR, np.dtype(np.float32), mode='nearest', order=1)
        bottom[0].diff[...] = interpolated_data
        
