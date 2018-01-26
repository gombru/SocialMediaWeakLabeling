import sys
import caffe
import numpy as np
from PIL import Image
import os

# Run in GPU
caffe.set_device(0)
caffe.set_mode_gpu()

# test = np.loadtxt('../../../datasets/Wikipedia/testset_txt_img_cat.list', dtype=str)
# with open('../../../datasets/MIRFLICKR25K/test_half.txt') as f:
with open('../../../datasets/MIRFLICKR25K/retrieval_list.txt') as f:
#with open('../../../datasets/PascalVOC2007/labels.txt') as f:
    test = f.readlines()


#Model name
model = 'SocialMedia_Inception_onlyFC_glove_tfidf_fromScratch_iter_2000'

#Output file
output_file_dir = '../../../datasets/MIRFLICKR25K/regression_output/' + model
if not os.path.exists(output_file_dir):
    os.makedirs(output_file_dir)
output_file_path = output_file_dir + '/test_half.txt'
output_file = open(output_file_path, "w")

# load net
net = caffe.Net('../googlenet_regression/prototxt/deploy.prototxt', '../../../datasets/SocialMedia/models/saved/'+ model + '.caffemodel', caffe.TEST)
# cats = ['art','biology','geography','history','literature','media','music','royalty','sport','warfare']

size = 227

# Reshape net
batch_size = 250 #300
net.blobs['data'].reshape(batch_size, 3, size, size)

print 'Computing  ...'

count = 0
i = 0
while i < len(test):
    indices = []
    if i % 100 == 0:
        print i

    # Fill batch
    for x in range(0, batch_size):

        if i > len(test) - 1: break

        # load image
        filename = '../../../datasets/MIRFLICKR25K/img/im' + str(int(test[i])) + '.jpg'
        im = Image.open(filename)
        im_o = im
        im = im.resize((size, size), Image.ANTIALIAS)
        indices.append(test[i])

        # Turn grayscale images to 3 channels
        if (im.size.__len__() == 2):
            im_gray = im
            im = Image.new("RGB", im_gray.size)
            im.paste(im_gray)

        #switch to BGR and substract mean
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= np.array((104, 117, 123))
        in_ = in_.transpose((2,0,1))

        net.blobs['data'].data[x,] = in_

        i += 1

    # run net and take scores
    net.forward()

    # Save results for each batch element
    for x in range(0,len(indices)):
        topic_probs = net.blobs['probs'].data[x]
        topic_probs_str = ''

        for t in topic_probs:
            topic_probs_str = topic_probs_str + ',' + str(t)

        output_file.write(str(int(indices[x])) + topic_probs_str + '\n')

output_file.close()

print "DONE"
print output_file_path


