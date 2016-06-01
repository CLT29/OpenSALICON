import numpy as np
from PIL import Image
import pdb
import matplotlib.pyplot as plt
import sys
import time
sys.path.insert(0, 'caffe/install/python') # UPDATE YOUR CAFFE PATH HERE
import caffe
caffe.set_mode_gpu()
caffe.set_device(1)
fine_imgs = []
coarse_imgs = []
fix_imgs = []
training_data_path = 'caffe/salicon/training_data/osie/' # PATH TO YOUR TRAINING DATA
MEAN_VALUE = np.array([103.939, 116.779, 123.68])   # BGR
MEAN_VALUE = MEAN_VALUE[:,None, None]
for i in range(1001, 1701):
    im = np.array(Image.open(training_data_path + 'coarse_images/' + str(i) + '.jpg'), dtype=np.float32) # in RGB
    # put channel dimension first
    im = np.transpose(im, (2,0,1))
    # switch to BGR
    im = im[::-1, :, :]
    # subtract mean
    im = im - MEAN_VALUE
    im = im[None,:]
    assert(im.shape == (1,3,600,800))
    # TEST - CONVERT TO DOUBLE
    im = im / 255
    im = im.astype(np.dtype(np.float32))
    coarse_imgs.append(im)
# now do the fine images
for i in range(1001, 1701):
    im = np.array(Image.open(training_data_path + 'fine_images/' + str(i) + '.jpg'), dtype=np.float32) # in RGB
    # put channel dimension first
    im = np.transpose(im, (2,0,1))
    # switch to BGR
    im = im[::-1, :, :]
    # subtract mean
    im = im - MEAN_VALUE
    im = im[None,:]
    assert(im.shape == (1,3,1200,1600))
    # TEST - CONVERT TO DOUBLE
    im = im / 255
    im = im.astype(np.dtype(np.float32))
    fine_imgs.append(im)
# load fixations
for i in range(1001, 1701):
    im = np.array(Image.open(training_data_path + 'fixation_images/' + str(i) + '.jpg'), dtype=np.float32)
    im = im[None,None,:]
    assert(im.shape == (1,1,38,50))
    # TEST - CONVERT TO DOUBLE
    im = im / 255
    im = im.astype(np.dtype(np.float32))
    fix_imgs.append(im)
assert(len(fix_imgs) == len(fine_imgs) and len(fine_imgs) == len(coarse_imgs))
assert(len(fix_imgs) == 700)
# load the solver
solver = caffe.SGDSolver('solver_new.prototxt')
solver.net.copy_from('salicon.caffemodel') # untrained.caffemodel
start_time = time.time()
idx_counter = 0
while time.time() - start_time < 43200:
    batch = np.random.permutation(len(fix_imgs))
    for i in range(0, len(batch)):
        idx_counter = idx_counter + 1
        print 'working on ' + str(i) + ' of ' + str(len(batch))
        fine_img_to_process = fine_imgs[batch[i]]
        coarse_img_to_process = coarse_imgs[batch[i]]
        fix_img_to_process = fix_imgs[batch[i]]
        solver.net.blobs['fine_scale'].data[...] = fine_img_to_process
        solver.net.blobs['coarse_scale'].data[...] = coarse_img_to_process
        solver.net.blobs['ground_truth'].data[...] = fix_img_to_process
        solver.step(1)
        if int(time.time() - start_time) % 10000 == 0:
            solver.net.save('train_output/finetuned_salicon_{}.caffemodel'.format(idx_counter))
    solver.net.save('train_output/finetuned_salicon_{}.caffemodel'.format(idx_counter))
