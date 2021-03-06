caffe_root = '../../caffe-master/'  # this file should be run from {caffe_root}/examples (otherwise change this line)

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
from create_solver import create_solver
from do_solve import do_solve
from pylab import *
import os


caffe.set_device(3)
caffe.set_mode_gpu()

weights = '../../../hd/datasets/SocialMedia/models/pretrained/bvlc_googlenet.caffemodel'
assert os.path.exists(weights)

niter = 5000000
base_lr = 0.001 #Starting from 0.01 (from quick solver) -- Working 0.001
display_interval = 1000 #400

#number of validating images  is  test_iters * batchSize
test_interval = 10000 #1000
test_iters = 600 #100

#Name for training plot and snapshots
training_id = 'instaFashion_GoogleNet_Contrastive_RandomNegative'

#Set solver configuration
solver_filename = create_solver('prototxt/trainval_instaFashion_all1loss.prototxt', 'prototxt/trainval_instaFashion_all1loss.prototxt', training_id, base_lr=base_lr)
#Load solver
solver = caffe.get_solver(solver_filename)

#Copy init weights
print("Initializing with ImageNet")
solver.net.copy_from(weights)

#Restore solverstate
#solver.restore('../../../datasets/SocialMedia/models/CNNRegression/instagram_cities_1M_Inception_frozen_500_chunck_iter_280000.solverstate')

print 'Running solver for %d iterations...' % niter
do_solve(niter, solver, display_interval, test_interval, test_iters, training_id)
print 'Done.'

