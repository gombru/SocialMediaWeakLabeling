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



class TripletLossLayer(caffe.Layer):
    global no_residual_list, margin, aux_idx

    def setup(self, bottom, top):
        """Setup the TripletDataLayer."""

        # assert shape(bottom[0].data) == shape(bottom[1].data)
        assert shape(bottom[1].data) == shape(bottom[2].data)

        params = eval(self.param_str)
        print params
        self.margin = params['margin']

        self.a = 1
        top[0].reshape(1)
        top[1].reshape(1)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        anchor_minibatch_db = []
        positive_minibatch_db = []
        negative_minibatch_db = []
        for i in range((bottom[0]).num):
            # anchor_minibatch_db.append(self.normalize(bottom[0].data[i]))
            # positive_minibatch_db.append(self.normalize(bottom[1].data[i]))
            # negative_minibatch_db.append(self.normalize(bottom[2].data[i]))
            anchor_minibatch_db.append(bottom[0].data[i])
            positive_minibatch_db.append(bottom[1].data[i])
            negative_minibatch_db.append(bottom[2].data[i])

        eq = 0
        loss = float(0)
        self.no_residual_list = []
        correct_pairs = 0
        # print "-------------Start Batch --> min(a): " + str(min(np.array(anchor_minibatch_db[0]))) + " /  a: " + str(np.array(anchor_minibatch_db[0][0:5]))

        self.aux_idx = 0
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

            # if i == 0:
            # print ('ap:' + str(ap) + ' ' + 'an:' + str(an) + ' ' + 'loss:' + str(_loss) + ' dist:' + str(dist) + ' cur_margin:' + str(ap - an))
            if _loss == 0: #
                correct_pairs += 1
                self.no_residual_list.append(i)
            elif sum(p) == 0 or sum(n) == 0: #or _loss > 1
                self.no_residual_list.append(i)
            elif ap == an or abs(ap-an) < 0.0001:
                self.no_residual_list.append(i)
                eq+=1


            loss += _loss

        loss = (loss / (2 * (bottom[0]).num))
        top[0].data[...] = loss
        top[1].data[...] = correct_pairs

        # print "----------------END Batch --> loss: " + str(loss) + ' --> eq = ' + str(eq) + 'correct_pairs = ' + str(correct_pairs)
        # if eq > 5:
        #     print "Sum(a): " + str(sum(a)) + " --> a " + str(a[0:5])
        #     time.sleep(30)


    def backward(self, top, propagate_down, bottom):
        considered_instances = bottom[0].num - len(self.no_residual_list)
        if propagate_down[0]:
            for i in range((bottom[0]).num):

                # if i !=0 and i == self.aux_idx:
                #     print "Aux idx found in BP: min(x_p): " + str(min(bottom[1].data[i])) + ' - ' + str(min(bottom[2].data[i]))

                if not i in self.no_residual_list:
                    # x_a = bottom[0].data[i]
                    x_p = bottom[1].data[i]
                    x_n = bottom[2].data[i]

                    # L2 normalization
                    # x_a = self.normalize(x_a)
                    # x_p = self.normalize(x_p)
                    # x_n = self.normalize(x_n)

                    # print x_a,x_p,x_n
                    # Raul. What is self.a? Is this gradient ok?
                    # Divided per batch size because Caffe doesn't average by default?
                    # bottom[0].diff[i] = self.a * ((x_n - x_p) / considered_instances)
                    bottom[0].diff[i] = self.a * ((x_n - x_p) / ((bottom[0]).num))
                    #bottom[1].diff[i] = self.a * ((x_p - x_a) / ((bottom[1]).num))
                    #bottom[2].diff[i] = self.a * ((x_a - x_n) / ((bottom[2]).num))

                else:
                    bottom[0].diff[i] = np.zeros(shape(bottom[0].data)[1])
                    #bottom[1].diff[i] = np.zeros(shape(bottom[1].data)[1])
                    #bottom[2].diff[i] = np.zeros(shape(bottom[2].data)[1])

                    # print 'select gradient_loss:',bottom[0].diff[0][0]
                    # print shape(bottom[0].diff),shape(bottom[1].diff),shape(bottom[2].diff)

            # print "BP: max:" +  str(np.max(bottom[0].diff[:,:])) + " min: " + str(np.min(bottom[0].diff[:,])) + " sum: " + str(sum(bottom[0].diff[:,:]))

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def normalize(self, v):
        norm = np.linalg.norm(v,2)
        if norm == 0:
            return v
        v = v /norm
        return v