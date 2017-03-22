import sys
import caffe
from deprocess import deprocess_net_image
from create_net import build_net
from pylab import *
import time

#Load weights of model to be evaluated
weights = 'models/trump/_iter_12000.caffemodel'
# weights = 'models/bvlc_reference_caffenet.caffemodel'


# Load style labels to style_labels
label_file = '../../../datasets/SocialMedia/lda_gt/trump/topic_names.txt'
labels = list(np.loadtxt(label_file, str, delimiter='\n'))

num_labels = 8
#Number of image to be tested are batch size (100) * test iterations
test_iters = 10
split_val = 'testTrumpUnique'
batch_size = 100
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


#Compute test accuracy
def eval_net(weights, test_iters):
    test_net = caffe.Net(build_net(split_val, num_labels, batch_size, resize_w, resize_h, resize_h, resize_h, crop_margin=0, mirror=0, rotate=0, HSV_prob=0, HSV_jitter=0, train=False), weights, caffe.TEST)
    accuracy = 0

    t = time.time()

    for it in xrange(test_iters):

        accuracy += test_net.forward()['acc']

        # Save accuracy per class
        for i in range(0, batch_size):
            probs = test_net.blobs['probs'].data[i]
            label = (-probs).argsort()[0]
            gt = int(test_net.blobs['label'].data[i])
            if label == gt:
                perClass[gt][0] += 1
            perClass[gt][1] += 1

    elapsed = time.time() - t

    accuracy /= test_iters

    return test_net, accuracy, elapsed

#Print global accuracy
test_net, accuracy, elapsed = eval_net(weights, test_iters)
print 'Accuracy: %3.1f%%' % (100*accuracy, )
print 'Elapsed time: ' + str(elapsed)


#Print accuracy per class
tot = 0
print '\nAccuracy per class: '
for i in range(0,num_labels):
    acc = perClass[i][0] / perClass[i][1]
    tot += acc
    print labels[i] + ' = ' + str(acc)

print '\nAccuracy averaged over classes: ' + str(float(tot) / num_labels)



#Print labels of an arbitrary image
batch_index = 0
image = test_net.blobs['data'].data[batch_index]
# plt.show("Fig 1")
# plt.imshow(deprocess_net_image(image))
print '\nActual label =', labels[int(test_net.blobs['label'].data[batch_index])]
disp_preds(test_net, image, labels, batch_index)

#Print labels of an arbitrary image
batch_index = 1
image = test_net.blobs['data'].data[batch_index]
# plt.show("Fig 2")
# plt.imshow(deprocess_net_image(image))
print '\nActual label =', labels[int(test_net.blobs['label'].data[batch_index])]
disp_preds(test_net, image, labels, batch_index)


