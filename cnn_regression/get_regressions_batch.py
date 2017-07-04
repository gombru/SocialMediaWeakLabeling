import sys
import caffe
import numpy as np
from PIL import Image
import os

# Run in GPU
caffe.set_device(0)
caffe.set_mode_gpu()

test = np.loadtxt('../../../datasets/SocialMedia/lda_gt/cities_instagram/testCitiesInstagram_1M_500_chunck_th0.txt', dtype=str)

#Model name
model = 'instagram_cities_1M_Inception_frozen_500_chunck_th0_multiGPU_iter_310000'

#Output file
output_file_dir = '../../../datasets/SocialMedia/regression_output/' + model
if not os.path.exists(output_file_dir):
    os.makedirs(output_file_dir)
output_file_path = output_file_dir + '/test.txt'
output_file = open(output_file_path, "w")

# load net
net = caffe.Net('../googlenet_regression/prototxt/deploy500.prototxt', '../../../datasets/SocialMedia/models/saved/'+ model + '.caffemodel', caffe.TEST)


size = 227

# Reshape net
batch_size = 300
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
        filename = '../../../datasets/SocialMedia/img_resized_1M/cities_instagram/' + test[i].split(',')[0] + '.jpg'
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
        in_ -= np.array((103.939, 116.779, 123.68))
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

        output_file.write(indices[x].split(',')[0] + topic_probs_str + '\n')

output_file.close()

print "DONE"


