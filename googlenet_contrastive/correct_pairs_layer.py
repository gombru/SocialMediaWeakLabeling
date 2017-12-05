# --------------------------------------------------------
# TRIPLET LOSS
# Copyright (c) 2015 Pinguo Tech.
# Written by David Lu
# --------------------------------------------------------

"""The data layer used during training a VGG_FACE network by triplet loss.
"""
import caffe
import numpy as np
from numpy import *
import time



class CorrectPairsLayer(caffe.Layer):
    global margin

    def setup(self, bottom, top):
        """Setup the TripletDataLayer."""

        # assert shape(bottom[0].data) == shape(bottom[1].data)
        assert shape(bottom[1].data) == shape(bottom[2].data)

        params = eval(self.param_str)
        self.margin = params['margin']

        self.a = 1
        top[0].reshape(1)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        anchor_minibatch_db = []
        positive_minibatch_db = []
        negative_minibatch_db = []
        for i in range((bottom[0]).num):
            anchor_minibatch_db.append(self.normalize(bottom[0].data[i]))
            positive_minibatch_db.append(self.normalize(bottom[1].data[i]))
            negative_minibatch_db.append(self.normalize(bottom[2].data[i]))

        correct_pairs = 0
        for i in range(((bottom[0]).num)):
            a = np.array(anchor_minibatch_db[i])
            p = np.array(positive_minibatch_db[i])
            n = np.array(negative_minibatch_db[i])
            a_p = a - p
            a_n = a - n
            ap = np.dot(a_p, a_p)
            an = np.dot(a_n, a_n)
            dist = (self.margin + ap - an)
            _loss = max(dist, 0.0)
            if _loss == 0:
                correct_pairs += 1

        top[0].data[...] = correct_pairs


    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def normalize(self, v):
        norm = np.linalg.norm(v,2)
        if norm == 0:
            return v
        v = v /norm
        return v