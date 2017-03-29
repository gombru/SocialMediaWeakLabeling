
import caffe
from caffe import layers as L
from caffe import params as P

frozen_weight_param = dict(lr_mult=0, decay_mult=0)
frozen_bias_param = dict(lr_mult=0, decay_mult=0)

weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)

boosted_weight_param = dict(lr_mult=10, decay_mult=1)
boosted_bias_param = dict(lr_mult=20, decay_mult=0)

learned_param = [weight_param, bias_param]
boosted_param = [boosted_weight_param, boosted_bias_param]
froozen_param = [frozen_weight_param, frozen_bias_param]



def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param,
              weight_filler=dict(type='gaussian', std=0.01),
              bias_filler=dict(type='constant', value=0.1)):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout, param=learned_param,
            weight_filler=dict(type='gaussian', std=0.005),
            bias_filler=dict(type='constant', value=0.1)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def ave_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.AVE, kernel_size=ks, stride=stride)

def build_net(split, num_classes, batch_size, resize_w, resize_h, crop_w=0, crop_h=0, crop_margin=0, mirror=0, rotate=0, HSV_prob=0, HSV_jitter=0, train=True):
    weight_param = dict(lr_mult=1, decay_mult=1)
    bias_param = dict(lr_mult=2, decay_mult=0)
    learned_param = [weight_param, bias_param]

    frozen_param = [dict(lr_mult=0)] * 2

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

    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, param=learned_param)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2, param=learned_param)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, param=learned_param)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, param=learned_param)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, param=learned_param)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=boosted_param) #4096
    if train:
        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
    else:
        fc7input = n.relu6
    n.fc7, n.relu7 = fc_relu(fc7input, 4096, param=boosted_param) #4096
    if train:
        n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True)
    else:
        fc8input = n.relu7

    fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=boosted_param)

    n.__setattr__('classifier', fc8)
    if not train:
        n.probs = L.Softmax(fc8)

    n.loss = L.SigmoidCrossEntropyLoss(fc8, n.label)
    # n.acc = L.MULTI_LABEL_ACCURACY(fc8, n.label)

    if train:
        with open('train.prototxt', 'w') as f:
            f.write(str(n.to_proto()))
            return f.name
    else:
        with open('val.prototxt', 'w') as f:
            f.write(str(n.to_proto()))
            return f.name