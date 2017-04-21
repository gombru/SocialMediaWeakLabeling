import caffe

import numpy as np
from PIL import Image
from PIL import ImageOps
import time

import random


class twoHeadsDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from the SBDD extended labeling
    of PASCAL VOC for semantic segmentation
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - sbdd_dir: path to SBDD `dataset` dir
        - split: train / seg11valid
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for SBDD semantic segmentation.

        N.B.segv11alid is the set of segval11 that does not intersect with SBDD.
        Find it here: https://gist.github.com/shelhamer/edb330760338892d511e.

        example

        params = dict(sbdd_dir="/path/to/SBDD/dataset",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="valid")
        """
        # config
        params = eval(self.param_str)
        self.dir = params['dir']
        self.train = params['train']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.batch_size = params['batch_size']
        self.resize_w = params['resize_w']
        self.resize_h = params['resize_h']
        self.crop_w = params['crop_w']
        self.crop_h = params['crop_h']
        self.crop_margin = params['crop_margin']
        self.mirror = params['mirror']
        self.rotate = params['rotate']
        self.HSV_prob = params['HSV_prob']
        self.HSV_jitter = params['HSV_jitter']

        self.num_classes= params['num_classes']

        self.cities = ['london', 'newyork', 'sydney', 'losangeles', 'chicago', 'melbourne', 'miami', 'toronto', 'singapore','sanfrancisco']

        # two tops: data and label
        if len(top) != 3:
            raise Exception("Need to define two tops: data, regression labels and classification label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/{}.txt'.format(self.dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        # self.labels = [int(i.split(',', 1)[1]) for i in self.indices]

        # Load labels for multiclass
        self.labels = np.zeros((len(self.indices), self.num_classes))
        self.labels_class = np.zeros((len(self.indices), 1))

        for c,i in enumerate(self.indices):
            data = i.split(',')
            #Load regression labels
            for l in range(0,self.num_classes):
                self.labels[c,l] = float(data[l+1])
            #Load classification labels
                self.labels_class[c] = self.cities.index(i.split('/')[0])


        self.indices = [i.split(',', 1)[0] for i in self.indices]

        # make eval deterministic
        # if 'train' not in self.split and 'trainTrump' not in self.split:
        #     self.random = False

        self.idx = np.arange(self.batch_size)
        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            for x in range(0,self.batch_size):
                self.idx[x] = random.randint(0, len(self.indices) - 1)
        else:
            for x in range(0, self.batch_size):
                self.idx[x] = x


        # reshape tops to fit
        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(self.batch_size, 3, params['crop_w'], params['crop_h'])
        top[1].reshape(self.batch_size, self.num_classes)
        top[2].reshape(self.batch_size, 1)


    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = np.zeros((self.batch_size, 3, self.crop_w, self.crop_h))
        self.label = np.zeros((self.batch_size, self.num_classes))
        self.label_class = np.zeros((self.batch_size, 1))

        for x in range(0,self.batch_size):
            self.data[x,] = self.load_image(self.indices[self.idx[x]])
            self.label[x,] = self.labels[self.idx[x],]
            self.label_class[x,] = self.labels_class[self.idx[x],]

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        top[2].data[...] = self.label_class

        self.idx = np.arange(self.batch_size)

        # pick next input
        if self.random:
            for x in range(0,self.batch_size):
                self.idx[x] = random.randint(0, len(self.indices) - 1)

        else:
            for x in range(0,self.batch_size):
                self.idx[x] = self.idx[x] + self.batch_size

            if self.idx[self.batch_size-1] == len(self.indices):
                for x in range(0, self.batch_size):
                    self.idx[x] = x


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        # print '{}/img/trump/{}.jpg'.format(self.dir, idx)
        # start = time.time()
        im = Image.open('{}/img_resized/cities_instagram/{}.jpg'.format(self.dir, idx))
        # To resize try im = scipy.misc.imresize(im, self.im_shape)
        #.resize((self.resize_w, self.resize_h), Image.ANTIALIAS) # --> No longer suing this resizing, no if below
        # end = time.time()
        # print "Time load and resize image: " + str((end - start))

        if im.size[0] != self.resize_w or im.size[1] != self.resize_h:
            im = im.resize((self.resize_w, self.resize_h), Image.ANTIALIAS)

        if( im.size.__len__() == 2):
            im_gray = im
            im = Image.new("RGB", im_gray.size)
            im.paste(im_gray)

        # start = time.time()
        if self.train: #Data Aumentation
            if(self.rotate is not 0):
                im = self.rotate_image(im)

            if self.crop_h is not self.resize_h or self.crop_h is not self.resize_h:
                im = self.random_crop(im)

            if(self.mirror and random.randint(0, 1) == 1):
                im = self.mirror_image(im)

            if(self.HSV_prob is not 0):
                im = self.saturation_value_jitter_image(im)

        # end = time.time()
        # print "Time data aumentation: " + str((end - start))

        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_



    #DATA AUMENTATION

    def random_crop(self,im):
        # Crops a random region of the image that will be used for training. Margin won't be included in crop.
        margin = 2
        left = random.randint(margin,self.resize_w - self.crop_w - 1 - margin)
        top = random.randint(margin,self.resize_h - self.crop_h - 1 - margin)
        im = im.crop((left, top, left + self.crop_w, top + self.crop_h))
        return im

    def mirror_image(self, im):
        return ImageOps.mirror(im)

    def rotate_image(self, im):
        return im.rotate(random.randint(-self.rotate, self.rotate))

    def saturation_value_jitter_image(self,im):
        if(random.randint(0, int(1/self.HSV_prob)) == 0):
            return im
        im = im.convert('HSV')
        data = np.array(im)  # "data" is a height x width x 3 numpy array
        data[:, :, 1] = data[:, :, 1] * random.uniform(1 - self.HSV_jitter, 1 + self.HSV_jitter)
        data[:, :, 2] = data[:, :, 2] * random.uniform(1 - self.HSV_jitter, 1 + self.HSV_jitter)
        im = Image.fromarray(data, 'HSV')
        im = im.convert('RGB')
        return im