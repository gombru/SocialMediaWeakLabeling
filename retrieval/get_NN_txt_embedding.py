import caffe
import numpy as np

def load_net(model_name,prototxt):
    # Run in GPU
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(prototxt, model_name, caffe.TEST)
    return net

def get_NN_txt_embedding(net,txt_embedding):
    net.blobs['label_p'].data[0,] = txt_embedding[...,np.newaxis]
    net.forward()
    NN_txt_embedding = net.blobs['TXT_FC_2'].data[0]
    return NN_txt_embedding

