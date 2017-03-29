import sys
import caffe
from deprocess import deprocess_net_image
from create_net import build_net
from pylab import *
import time

#Load weights of model to be evaluated
weights = 'models/trump/_iter_200.caffemodel'
# weights = 'models/bvlc_reference_caffenet.caffemodel'


# Load style labels to style_labels
label_file = '../../../datasets/SocialMedia/lda_gt/trump/topic_names.txt'
labels = list(np.loadtxt(label_file, str, delimiter='\n'))

num_labels = 4
#Number of image to be tested are batch size (100) * test iterations
test_iters = 1
split_val = 'valReg'
batch_size = 5
resize_w = 227
resize_h = 227

#Print per class accuracy of last batch
perClass = np.zeros([len(labels),2])

def disp_preds(net, image, labels, batch_index, k=5):
    input_blob = net.blobs['data']
    net.blobs['data'].data[0, ...] = image
    # probs = net.forward(start='conv1')['probs'][0]
    probs = net.blobs['probs'].data[batch_index]
    top_k = (-probs).argsort()[:k]
    print 'top %d predicted labels =' % (k)
    print '\n'.join('\t(%d) %5.2f%% %s' % (i+1, 100*probs[p], labels[p])
                    for i, p in enumerate(top_k))



test_net = caffe.Net(build_net(split_val, num_labels, batch_size, resize_w, resize_h, resize_h, resize_h, crop_margin=0, mirror=0, rotate=0, HSV_prob=0, HSV_jitter=0, train=False), weights, caffe.TEST)
loss = 0
for it in xrange(test_iters):

    loss += test_net.forward()['loss']

#Print labels of an arbitrary image
batch_index = 0
image = test_net.blobs['data'].data[batch_index]
# plt.show("Fig 1")
# plt.imshow(deprocess_net_image(image))
print '\nActual label =', test_net.blobs['label'].data[batch_index]
disp_preds(test_net, image, labels, batch_index)

#Print labels of an arbitrary image
batch_index = 1
image = test_net.blobs['data'].data[batch_index]
# plt.show("Fig 2")
# plt.imshow(deprocess_net_image(image))
print '\nActual label =', test_net.blobs['label'].data[batch_index]
disp_preds(test_net, image, labels, batch_index)

#Print labels of an arbitrary image
batch_index = 2
image = test_net.blobs['data'].data[batch_index]
# plt.show("Fig 2")
# plt.imshow(deprocess_net_image(image))
print '\nActual label =', test_net.blobs['label'].data[batch_index]
disp_preds(test_net, image, labels, batch_index)

# Print labels of an arbitrary image
batch_index = 3
image = test_net.blobs['data'].data[batch_index]
# plt.show("Fig 2")
# plt.imshow(deprocess_net_image(image))
print '\nActual label =', test_net.blobs['label'].data[batch_index]
disp_preds(test_net, image, labels, batch_index)

# Print labels of an arbitrary image
batch_index = 4
image = test_net.blobs['data'].data[batch_index]
# plt.show("Fig 2")
# plt.imshow(deprocess_net_image(image))
print '\nActual label =', test_net.blobs['label'].data[batch_index]
disp_preds(test_net, image, labels, batch_index)




