caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
from PIL import Image


# Run in GPU
caffe.set_device(0)
caffe.set_mode_gpu()

test = np.loadtxt('../../../SocialMedia/testCities1day.txt', dtype=str)

#Output file
output_file_path = '../../../datasets/SocialMedia/regression_output/testCitiesClassification.txt'
output_file = open(output_file_path, "w")

# load net
net = caffe.Net('VGG_ILSVRC_16_layers_deploy.prototxt', '../../../datasets/SocialMedia/models/CNNRegression/train_iter_100000.caffemodel', caffe.TEST)


print 'Computing  ...'

count = 0

for idx in test:

    count = count + 1
    if count % 100 == 0:
        print count

    # load image
    im = Image.open(idx)
    im_o = im
    im = im.resize((224, 224), Image.ANTIALIAS)

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


    net.blobs['data'].data[...] = in_

    # run net and take scores
    net.forward()

    # Compute SoftMax HeatMap
    topic_probs = net.blobs['fc8'].data[0]   #Text score

    topic_probs_str = ''
    for t in topic_probs:
        topic_probs_str = topic_probs_str + ',' + t

    output_file.write(idx + topic_probs_str + '\n')

output_file.close()



