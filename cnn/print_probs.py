caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


# Run in GPU
caffe.set_device(0)
caffe.set_mode_gpu()

test = np.loadtxt('../../../datasets/SocialMedia/londonTestImages.txt', dtype=str)


# load net
net = caffe.Net('deploy.prototxt', '../../../datasets/SocialMedia/models/CNN/intagram_cities_VGG16__iter_21600.caffemodel', caffe.TEST)

# Load style labels to style_labels
label_file = '../../../datasets/SocialMedia/lda_gt/cities_instagram/topic_names.txt'
labels = list(np.loadtxt(label_file, str, delimiter='\n'))


print 'Computing  ...'

count = 0
k = 5

for idx in test:
    idx = idx.split(',')[0]

    count = count + 1
    if count % 100 == 0:
        print count

    # load image
    im = Image.open('../../../datasets/SocialMedia/img_resized/cities_instagram/' + idx + '.jpg')
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
    probs = net.blobs['probs'].data[0]
    top_k = (-probs).argsort()[:k]
    # print 'top %d predicted labels =' % (k)
    # print 'top %d predicted labels =' % (k)
    text = '\n'.join('\t(%d) %5.2f%% %s' % (i+1, 100*probs[p], labels[p]) for i, p in enumerate(top_k))

    draw = ImageDraw.Draw(im_o)
    #font = ImageFont.truetype(<font-file>, <font-size>)
    font = ImageFont.truetype("/usr/share/fonts/truetype/droid/DroidSans-Bold.ttf", 20)
    lines = text.split('\n')
    y = 0
    for l in lines:
        y += 20
        draw.text((0, y),l,(10,255,10), font=font)

    if top_k[0] == 0:
        im_o.save('../../../datasets/SocialMedia/samples/' + idx.replace('london','london_TP') + '.jpg')
    im_o.save('../../../datasets/SocialMedia/samples/' + idx + '.jpg')



