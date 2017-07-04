
'''
Creates the data layer. It creates a data_layer.protxt that it is not used. Its content is the data layer calling code
and should be copied into the train.prototxt and val.protxt
TODO: Check how Caffe interacts with the class of the created layer
'''

import caffe
from caffe import layers as L

split_train = 'lda_gt/train_500_chunck80000'
split_val = 'lda_gt/myval_500_chunck80000'
dir = '../../../datasets/WebVision'

num_labels = 500
batch_size = 100 #AlexNet 100, VGG 40
resize = False #Resize the image to the given size before cropping
resize_w = 224
resize_h = 224
crop_w = 224 #Train with a random crop of this size
crop_h = 224 #227 AlexNet, 224 VGG16, Inception
crop_margin = 2 #The crop won't include the margin in pixels
mirror = True #Mirror images with 50% prob
rotate_prob = .2 #Rotation probability
rotation_angle = 8 #15,8 #Rotate with angle between -a and a
HSV_prob = .2 #0.3,0.15 #Jitter saturation and vale of the image with this prob
HSV_jitter = 0.05 #0.1,0.05 #Saturation and value will be multiplied by 2 different random values between 1 +/- jitter
color_casting_prob = 0.05 #0.1 #Alterates each color channel (with the given prob for each channel) suming jitter
color_casting_jitter = 10 #Sum/substract 10 from the color channel
scaling_prob = .5 #Rescale the image with the factor given before croping
scaling_factor = 1.3 #Rescaling factor

n = caffe.NetSpec()

pydata_params = dict(split=split_train, mean=(104, 117, 123))

pydata_params['dir'] = dir
pydata_params['train'] = True
pydata_params['num_classes'] = num_labels
pydata_params['batch_size'] = batch_size
pydata_params['resize'] = resize
pydata_params['resize_w'] = resize_w
pydata_params['resize_h'] = resize_h
pydata_params['crop_w'] = crop_w
pydata_params['crop_h'] = crop_h
pydata_params['crop_margin'] = crop_margin
pydata_params['mirror'] = mirror
pydata_params['rotate_prob'] = rotate_prob
pydata_params['rotate_angle'] = rotation_angle
pydata_params['HSV_prob'] = HSV_prob
pydata_params['HSV_jitter'] = HSV_jitter
pydata_params['color_casting_prob'] = color_casting_prob
pydata_params['color_casting_jitter'] = color_casting_jitter
pydata_params['scaling_prob'] = scaling_prob
pydata_params['scaling_factor'] = scaling_factor





pylayer = 'customDataLayer'

n.data, n.label = L.Python(module='layers', layer=pylayer,
                                          ntop=2, param_str=str(pydata_params))
with open('prototxt/data_layer.prototxt', 'w') as f:
        f.write(str(n.to_proto()))


