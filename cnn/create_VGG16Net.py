
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

def build_VGG16Net(split, num_classes, batch_size, resize_w, resize_h, crop_w=0, crop_h=0, crop_margin=0, mirror=0, rotate=0, HSV_prob=0, HSV_jitter=0, train=True):

    weight_param = dict(lr_mult=1, decay_mult=1)
    bias_param = dict(lr_mult=2, decay_mult=0)
    learned_param = [weight_param, bias_param]

    frozen_param = [dict(lr_mult=0)] * 2

    n = caffe.NetSpec()

    pydata_params = dict(split=split, mean=(103.939, 116.779, 123.68))

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

    pylayer = 'customDataLayer'

    n.data, n.label = L.Python(module='layers', layer=pylayer,
                               ntop=2, param_str=str(pydata_params))



    # conv
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 3, 64, pad=1, param=learned_param)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 3, 64, pad=1, param=learned_param)
    n.pool1 = max_pool(n.relu1_2, 2, stride=2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 3, 128, pad=1, param=learned_param)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 3, 128, pad=1, param=learned_param)
    n.pool2 = max_pool(n.relu2_2, 2, stride=2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 3, 256, pad=1, param=learned_param)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 3, 256, pad=1, param=learned_param)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 3, 256, pad=1, param=learned_param)
    n.pool3 = max_pool(n.relu3_3, 2, stride=2)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 3, 512, pad=1, param=learned_param)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 3, 512, pad=1, param=learned_param)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 3, 512, pad=1, param=learned_param)
    n.pool4 = max_pool(n.relu4_3, 2, stride=2)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 3, 512, pad=1, param=learned_param)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 3, 512, pad=1, param=learned_param)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 3, 512, pad=1, param=learned_param)
    n.pool5 = max_pool(n.relu5_3, 2, stride=2)

    # fully conn
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=boosted_param)
    if train:
        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True, dropout_ratio=0.5) #0.5
    else:
        fc7input = n.relu6

    n.fc7, n.relu7 = fc_relu(fc7input, 4096, param=boosted_param)
    if train:
        n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True, dropout_ratio=0.5) #0.5
    else:
        fc8input = n.relu7

    fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=boosted_param)

    n.__setattr__('classifier', fc8)
    if not train:
        n.probs = L.Softmax(fc8)

    n.loss = L.SoftmaxWithLoss(fc8, n.label)
    n.acc = L.Accuracy(fc8, n.label)





    if train:
        with open('train.prototxt', 'w') as f:
            f.write(str(n.to_proto()))
            return f.name
    else:
        with open('val.prototxt', 'w') as f:
            f.write(str(n.to_proto()))
            return f.name