
'''
Creates the data layer. It creates a data_layer.protxt that it is not used. Its content is the data layer calling code
and should be copied into the train.prototxt and val.protxt
TODO: Check how Caffe interacts with the class of the created layer
'''

import caffe
from caffe import layers as L

split_train = 'trainCitiesInstagram40'
split_val = 'valCitiesInstagram40'
num_labels = 40
batch_size = 100 #AlexNet 100, VGG 40
resize_w = 300
resize_h = 300
crop_w = 224 #227 AlexNet, 224 VGG16, Inception
crop_h = 224
crop_margin = 10 #The crop won't include the margin in pixels
mirror = True #Mirror images with 50% prob
rotate = 0 #15,8 #Always rotate with angle between -a and a
HSV_prob = 0 #0.3,0.15 #Jitter saturation and vale of the image with this prob
HSV_jitter = 0 #0.1,0.05 #Saturation and value will be multiplied by 2 different random values between 1 +/- jitter

def build_net(split, num_classes, batch_size, resize_w, resize_h, crop_w=0, crop_h=0, crop_margin=0, mirror=0, rotate=0, HSV_prob=0, HSV_jitter=0, train=True):

    n = caffe.NetSpec()

    pydata_params = dict(split=split, mean=(104.00699, 116.66877, 122.67892))

    pydata_params['dir'] = '../../../datasets/SocialMedia'
    pydata_params['train'] = train
    pydata_params['batch_size'] = batch_size
    pydata_params['resize_w'] = resize_w
    pydata_params['resize_h'] = resize_h
    pydata_params['crop_w'] = crop_w
    pydata_params['crop_h'] = crop_h
    pydata_params['crop_margin'] = crop_margin
    pydata_params['mirror'] = mirror
    pydata_params['rotate'] = rotate
    pydata_params['HSV_prob'] = HSV_prob
    pydata_params['HSV_jitter'] = HSV_jitter
    pydata_params['num_classes'] = num_classes


    pylayer = 'customDataLayer'

    n.data, n.label = L.Python(module='layers', layer=pylayer,
                               ntop=2, param_str=str(pydata_params))

    with open('data_layer.prototxt', 'w') as f:
            f.write(str(n.to_proto()))


#JUSTO TO CREATE DATA LAYER; NET ARQ IS HARDCODED
build_net(split_train, num_labels, batch_size, resize_w, resize_h, crop_w, crop_h, crop_margin, mirror, rotate, HSV_prob, HSV_jitter, train=True)