# Slides a window across an image to get a heatmap with windowSize resolution showing which images windows activate the net for a certain category

import sys
sys.path.insert(0, '../cnn')
import caffe
import numpy as np
from PIL import Image
from pylab import plt
from matplotlib import cm
from PIL import ImageFont
from PIL import ImageDraw

# Run in GPU
caffe.set_device(0)
caffe.set_mode_gpu()

windowSize = 40
stepSize = 10
# load net
net = caffe.Net('../cnn/deploy.prototxt', '../../../datasets/SocialMedia/models/saved/intagram_cities_VGG16__iter_21600.caffemodel', caffe.TEST)

test = np.loadtxt('../../../datasets/SocialMedia/testCitiesClassification.txt', dtype=str)

cities = ['london','newyork','sydney','losangeles','chicago','melbourne','miami','toronto','singapore','sanfrancisco']


def preprocess(im):
    # load image
    # filename = '../../../datasets/SocialMedia/img_resized/cities_instagram/' + idx.split(',')[0] + '.jpg'

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
    gt = idx.split('/')[0]

    im = Image.open(filename)
    im = im.resize((224, 224), Image.ANTIALIAS)
    im_o = im.copy()

    im = preprocess(im)
    out_img = np.zeros(im[0,:,:].shape)

    # Infer whole image
    probs = infer(im)
    top_topic = probs.argmax()

    for (x, y, window) in sliding_window(im,stepSize=stepSize,windowSize=(windowSize,windowSize)):
        # occluded = np.zeros(im.shape)
        # occluded[:,y:y+windowSize,x:x+windowSize] = window

        occluded = im.copy()
        occluded[:, y:y + windowSize, x:x + windowSize] = 0

        probs = infer(occluded)
        value = probs[top_topic]
        # if value > 0.2:
        # print value
        out_img[y:y + windowSize, x:x + windowSize] =  np.maximum(out_img[y:y + windowSize, x:x + windowSize], value)
        # plt.imshow(np.array(im_o, dtype=np.uint8)[y:y + windowSize, x:x + windowSize])
        # plt.imshow(out_img)


    image = out_img.astype(np.float32)  # convert to float
    image -= image.min()  # ensure the minimal value is 0.0
    image /= image.max()
    image = Image.fromarray(np.uint8(cm.rainbow(image) * 255))
    colormap = np.array(image, dtype=np.uint8)
    im_o = np.array(im_o, dtype=np.uint8)

    # Stack images horitzontaly and save
    imgs_comb = np.hstack([im_o,colormap[:,:,0:3]])
    imgs_comb = Image.fromarray(imgs_comb)

    #Draw GT class
    draw = ImageDraw.Draw(imgs_comb)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    font = ImageFont.truetype("/usr/share/fonts/truetype/droid/DroidSans-Bold.ttf", 20)
    draw.text((0, 0), 'GT: ' + gt, (10, 255, 10), font=font)
    draw.text((0, 20), 'OUT: ' + cities[top_topic], (10, 255, 10), font=font)


    imgs_comb.save('../../../datasets/SocialMedia/heatmaps/' + filename.split('/')[-1])
    print filename

print 'DONE'





