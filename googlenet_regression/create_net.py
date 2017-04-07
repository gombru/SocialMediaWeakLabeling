#DEFINING AND RUNNING THE NET

caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe import layers as L
from caffe import params as P
import tempfile



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
