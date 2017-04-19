# Slides a window across an image to get a heatmap with windowSize resolution showing which images windows activate the net for a certain category

import sys
sys.path.insert(0, '../cnn')
import caffe
import numpy as np
from PIL import Image
from pylab import plt

# Run in GPU
caffe.set_device(0)
caffe.set_mode_gpu()

windowSize = 40
stepSize = 10
# load net
net = caffe.Net('CaffeNet_deploy.prototxt', '../../../datasets/SocialMedia/models/saved/intagram_cities_CaffeNet_iter_40000.caffemodel', caffe.TEST)

test = np.loadtxt('../../../datasets/SocialMedia/testCitiesInstagram40.txt', dtype=str)

def preprocess(filename):
    # load image
    # filename = '../../../datasets/SocialMedia/img_resized/cities_instagram/' + idx.split(',')[0] + '.jpg'
    im = Image.open(filename)
    im = im.resize((227, 227), Image.ANTIALIAS)

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

    return in_

def infer(image):
    net.blobs['data'].data[...] = image
    # run net and take scores
    net.forward()
    # Compute SoftMax HeatMap
    topic_probs = net.blobs['probs'].data[0]   #Text score
    return topic_probs


def deprocess_net_image(image):

    import numpy as np

    image = image.copy()              # don't modify destructively
    image = image[::-1]               # BGR -> RGB
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    image += [0, 117, 104]          # (approximately) undo mean subtraction  Don't redo red!

    # clamp values in [0, 255]
    image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image


def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[1], stepSize):
		for x in xrange(0, image.shape[2], stepSize):
			# yield the current window
			yield (x, y, image[:,y:y + windowSize[1], x:x + windowSize[0]])

for idx in test:

    filename = '../../../datasets/SocialMedia/img_resized/cities_instagram/' + idx.split(',')[0] + '.jpg'
# filename = '../../../datasets/SocialMedia/img_resized/cities_instagram/london/1481113621266466265.jpg'
    img = preprocess(filename)
    out_img = np.zeros(img.shape)
    out_img[1,:,:] = (img[0,:,:] + img[1,:,:] + img[2,:,:]) / 3
    out_img[0,:,:] = (img[0,:,:] + img[1,:,:] + img[2,:,:]) / 3

    # Infer whole image
    probs = infer(img)
    top_topic = probs.argmax()

    for (x, y, window) in sliding_window(img,stepSize=stepSize,windowSize=(windowSize,windowSize)):
        occluded = np.zeros(img.shape)
        occluded[:,y:y+windowSize,x:x+windowSize] = window
        # plt.imshow(deprocess_net_image(occluded))

        probs = infer(occluded)
        value = probs[top_topic]
        if value > float(1) / 40:
            out_img[2, y:y + windowSize, x:x + windowSize] =  np.maximum(out_img[2, y:y + windowSize, x:x + windowSize], out_img[1, y:y + windowSize, x:x + windowSize] + value * 254 * 10)
            plt.imshow(deprocess_net_image(occluded))
            print value
        plt.imshow(deprocess_net_image(out_img))

    heatmap = Image.fromarray(deprocess_net_image(out_img))
    heatmap.save('../../../datasets/SocialMedia/heatmaps/' + filename.split('/')[-1])
    print filename







