
import sys
import caffe
from create_AlexNet import build_AlexNet
from create_AlexNet_2heads import build_AlexNet_2heads
from create_VGG16Net import build_VGG16Net
from create_solver import create_solver
from do_solve import do_solve
import os


caffe.set_device(0)
caffe.set_mode_gpu()

weights = '../../../datasets/SocialMedia/models/pretrained/bvlc_reference_caffenet.caffemodel'
assert os.path.exists(weights)


split_train = 'trainCitiesInstagram100'
split_val = 'valCitiesInstagram100'
num_labels = 100
batch_size = 256 #AlexNet 100, VGG 40
resize_w = 300
resize_h = 300
crop_w = 227 #227 AlexNet, 224 VGG16
crop_h = 227
crop_margin = 10 #The crop won't include the margin in pixels
mirror = True #Mirror images with 50% prob
rotate = 0 #15,8 #Always rotate with angle between -a and a
HSV_prob = 0 #0.3,0.15 #Jitter saturation and vale of the image with this prob
HSV_jitter = 0 #0.1,0.05 #Saturation and value will be multiplied by 2 different random values between 1 +/- jitter


#Create the net architecture
net_train = build_AlexNet_2heads(split_train, num_labels, batch_size, resize_w, resize_h, crop_w, crop_h, crop_margin, mirror, rotate, HSV_prob, HSV_jitter, train=True)
#Prepare validation net
net_val = build_AlexNet_2heads(split_val, num_labels, batch_size, crop_w, crop_h, crop_h, crop_h, 0, 0, 0, 0, 0, train=False)


base_lr = 0.001 #VGG 0.0001  #AlexNet 0.0001 #Regression 0.001

display_interval = 20
niter = 100011100

#number of validating images  is  test_iters * batchSize
test_interval = 120 #200
test_iters = 20 #20

#Set solver configuration
solver_filename = create_solver(net_train, net_val, base_lr=base_lr)
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