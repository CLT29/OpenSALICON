# OpenSALICON

OpenSALICON is an open source implementation of the SALICON saliency model cited below. 
> Huang, X., Shen, C., Boix, X., & Zhao, Q. (2015). SALICON: Reducing the Semantic Gap in Saliency Prediction by Adapting Deep Neural Networks. In Proceedings of the IEEE International Conference on Computer Vision (pp. 262-270).

### Citing OpenSALICON

If you find OpenSALICON useful in your research, please cite:
````
@TECHREPORT {christopherleethomas2016,
    author      = {Christopher Lee Thomas},
    title       = {OpenSalicon: An Open Source Implementation of the Salicon Saliency Model},
    institution = {University of Pittsburgh},
    year        = {2016},
    number      = {TR-2016-02}
}
````
    
### Using OpenSALICON

##### Requirements: software
1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  ```
2. Python packages you might not have: `numpy`, `PIL`, `matplotlib`, `scipy` 

##### Basic Demo (producing saliency maps from images)
1. Edit the `Salicon.py` file to include the path to your caffe python install where indicated (i.e. update the line below to the path containing your caffe installation):
    ````python
        sys.path.insert(0, 'caffe/install/python') # PATH TO CAFFE PYTHON INSTALL
    ````
    Additionally, depending on whether you are using a GPU or not, you may need to edit the following two lines:
    ````python
    caffe.set_mode_gpu()
    caffe.set_device(1)
    ````
    If you do not have a CPU, be sure to remove the ````set_device(1) ```` line and change the mode selection to ```` caffe.set_mode_cpu() ````. Depending on your setup, you may need to change the device ID from 1 to 0 or some other number depending on your hardware configuration.
2. Download the caffemodel files using **[this link](http://www.cs.pitt.edu/~chris/files/2016/model_files.tgz)**
3. The following commands should allow you to compute the saliency map for an image using the pretrained model. **IMPORTANT:** When using on your own data, ensure that your data is in 256 RGB format. If it is not, you will need to manually do some pre-processing first.
4. Run python and execute the following commands:
    ````python
    from Salicon import Salicon
    sal = Salicon()
    map = sal.compute_saliency('face.jpg')
    # map contains saliency map in double format.
    ````
    ![Original Image](/face.jpg "Original Image")
    ![Resulting Map](/face_map.jpg "Resulting Map")
5. You can then perform thresholding (if you prefer) on the output or use it directly.

##### Training your own model

Use the finetune_salicon.py file to train your own model. See our technical report for more details on how to do this. We provide two solvers ```` solver.prototxt ```` and ```` solver_new.prototxt ```` for this purpose. Solver_new attempts to use ADADELTA to adjust the learning rate dynamically. ````finetune_salicon.prototxt```` provides a model definition file for our implementation in which you can adjust learning rates per layer and more customization. Our ```` finetune_salicon.py```` file provides the basic functionality of reading in the input images and fixation maps, initializing the solver, loading the data in batches, solving the model, and saving the results. You may need to adjust the solver to your particular dataset.

#### Parting Words

OpenSALICON performs on-par with the results produced by the official SALICON demo website when compared using AUC scores to ground truth human fixations. If you have any questions, discover any bugs, or make improvements to this code, please feel free to create an issue or push request.
