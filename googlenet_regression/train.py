caffe_root = '../../caffe-master/'  # this file should be run from {caffe_root}/examples (otherwise change this line)

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
from create_solver import create_solver
from do_solve import do_solve
from pylab import *
import os


caffe.set_device(0)
caffe.set_mode_gpu()

weights = '../../../datasets/WebVision/models/saved/WebVision_Inception_frozen_word2vec_tfidfweighted_divbymax_iter_460000.caffemodel'
assert os.path.exists(weights)

niter = 10001111
base_lr = 0.001 #Starting from 0.01 (from quick solver) -- Working 0.001
display_interval = 5 #200

#number of validating images  is  test_iters * batchSize
test_interval = 100 #1000
test_iters = 1 #80

#Name for training plot and snapshots
training_id = 'mirflickr_Inception_frozen_word2vec_mean_finetuned_5000lrdecrease_half'

#Set solver configuration
solver_filename = create_solver('prototxt/train_frozen_word2vec_mean_mirflickr.prototxt', 'prototxt/val_frozen_word2vec_mean_mirflickr.prototxt', training_id, base_lr=base_lr)
#Load solver
solver = caffe.get_solver(solver_filename)

#Copy init weights
solver.net.copy_from(weights)

#Restore solverstate
#solver.restore('../../../datasets/SocialMedia/models/CNNRegression/instagram_cities_1M_Inception_frozen_500_chunck_iter_280000.solverstate')


print 'Running solvers for %d iterations...' % niter
solvers = [('my_solver', solver)]
_, _, _ = do_solve(niter, solvers, display_interval, test_interval, test_iters, training_id)
print 'Done.'

