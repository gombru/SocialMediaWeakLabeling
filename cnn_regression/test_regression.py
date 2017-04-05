import sys
import caffe
from deprocess import deprocess_net_image
from create_VGG16Net import build_VGG16Net
from pylab import *
import time

#Load weights of model to be evaluated
weights = '../../../datasets/SocialMedia/models/CNNRegression/intagram_cities_VGG16__iter_96000.caffemodel'
# weights = 'models/bvlc_reference_caffenet.caffemodel'


# Load style labels to style_labels
# label_file = '../../../datasets/SocialMedia/lda_gt/trump/topic_names.txt'
# labels = list(np.loadtxt(label_file, str, delimiter='\n'))

num_labels = 100
#Number of image to be tested are batch size (100) * test iterations
test_iters = 1
split_val = 'minitrainCitiesInstagram'
batch_size = 5
resize_w = 224
resize_h = 224
k = 5

#Print per class accuracy of last batch
perClass = np.zeros(num_labels)

def disp_preds(net, image, batch_index):
    input_blob = net.blobs['data']
    net.blobs['data'].data[0, ...] = image
    # probs = net.forward(start='conv1')['probs'][0]
    probs = net.blobs['probs'].data[batch_index]
    # print '\nPredic l. =', probs
    top_k = (-probs).argsort()[:k]
    print 'top %d predicted labels =' % (k)
    print '\n'.join('\t(%d) %5.2f%% %s' % (i+1, 100*probs[p], p)
                    for i, p in enumerate(top_k))



test_net = caffe.Net(build_VGG16Net(split_val, num_labels, batch_size, resize_w, resize_h, resize_h, resize_h, crop_margin=0, mirror=0, rotate=0, HSV_prob=0, HSV_jitter=0, train=False), weights, caffe.TEST)
loss = 0
for it in xrange(test_iters):

    loss += test_net.forward()['loss']

for b in range(0,5):
    #Print labels of an arbitrary image
    batch_index = b
    image = test_net.blobs['data'].data[batch_index]
    # plt.show("Fig 1")
    # plt.imshow(deprocess_net_image(image))
    gt = test_net.blobs['label'].data[batch_index]
    # print '\nActual label =', gt]
    top_k = (-gt).argsort()[:k]
    print 'top %d GT labels =' % (k)
    print '\n'.join('\t(%d) %5.2f%% %s' % (i + 1, 100 * gt[p], p)
                    for i, p in enumerate(top_k))
    disp_preds(test_net, image, batch_index)
    print '\n ________________________________ \n'

