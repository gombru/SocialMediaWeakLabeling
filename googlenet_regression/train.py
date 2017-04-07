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

weights = '../../../datasets/SocialMedia/models/pretrained/bvlc_googlenet.caffemodel'
assert os.path.exists(weights)

niter = 1000111
base_lr = 0.0001 #Starting from 0.01
display_interval = 2

#number of validating images  is  test_iters * batchSize
test_interval = 20
test_iters = 10

#Set solver configuration
solver_filename = create_solver('train.prototxt', 'val.prototxt', base_lr=base_lr)
#Load solver
solver = caffe.get_solver(solver_filename)

#Copy init weights
solver.net.copy_from(weights)

#Restore solverstate
#solver.restore('models/IIT5K/cifar10/IIT5K_iter_15000.caffemodel')


print 'Running solvers for %d iterations...' % niter
solvers = [('my_solver', solver)]
_, _, _ = do_solve(niter, solvers, display_interval, test_interval, test_iters)
print 'Done.'

