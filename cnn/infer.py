caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
from deprocess import deprocess_net_image
from create_net import build_net
from pylab import *
import time


#Load weights of model to be evaluated
weights = caffe_root + 'models/IIT5K/cnn-char-best/IIT5K_iter_15000.caffemodel'

# Load style labels to style_labels
label_file = caffe_root + 'data/IIT5K/char_names.txt'
labels = list(np.loadtxt(label_file, str, delimiter='\n'))

num_labels = 36
#Number of image to be tested are batch size (100) * test iterations
test_iters = 10
split_val = 'test'
batch_size = 100
resize_w = 32
resize_h = 32


#Compute test accuracy
def eval_net(weights, test_iters):
    test_net = caffe.Net(build_net(split_val, num_labels, batch_size, resize_w, resize_h, resize_h, resize_h, crop_margin=0, mirror=0, rotate=0, HSV_prob=0, HSV_jitter=0, train=False), weights, caffe.TEST)
    accuracy = 0

    t = time.time()

    for it in xrange(test_iters):
        accuracy += test_net.forward()['acc']
    elapsed = time.time() - t

    accuracy /= test_iters

    return test_net, accuracy, elapsed


test_net, accuracy, elapsed = eval_net(weights, test_iters)
print 'Accuracy, finetuned from ImageNet initialization: %3.1f%%' % (100*accuracy, )
print 'Elapsed time: ' + str(elapsed)

#Print labels of an arbitrary image
batch_index = 6
image = test_net.blobs['data'].data[batch_index]
plt.imshow(deprocess_net_image(image))
print 'actual label =', labels[int(test_net.blobs['label'].data[batch_index])]


def disp_preds(net, image, labels, k=5):
    input_blob = net.blobs['data']
    net.blobs['data'].data[0, ...] = image
    probs = net.forward(start='conv1')['probs'][0]
    top_k = (-probs).argsort()[:k]
    print 'top %d predicted labels =' % (k)
    print '\n'.join('\t(%d) %5.2f%% %s' % (i+1, 100*probs[p], labels[p])
                    for i, p in enumerate(top_k))

disp_preds(test_net, image, labels)

