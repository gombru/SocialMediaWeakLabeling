"""
Trains a model using one or more GPUs.
"""

from multiprocessing import Process
import caffe
import numpy as np
from pylab import zeros, arange, subplots, plt, savefig

training_id = 'instagram_cities_1M_Inception_frozen_500_chunck_multiGPU' # name to save the training plots
solver_path = 'solver_multiGPU.prototxt' # solver proto definition
snapshot = '../../../datasets/SocialMedia/models/pretrained/bvlc_googlenet.caffemodel' # snapshot to restore (only weights initialzation)
gpus = [0] # list of device ids # last GPU requires por mem (9000-5000)
timing = False # show timing info for compute and communications
plotting = True # plot loss
test_interval = 5000 # do validation each this iterations
test_iters = 200 # number of validation iterations


def train(solver_path,  snapshot,  gpus):
    # NCCL uses a uid to identify a session
    uid = caffe.NCCL.new_uid()

    caffe.init_log()
    print 'Using devices %s' % str(gpus)

    procs = []
    for rank in range(len(gpus)):
        p = Process(target=solve,
                    args=(solver_path, snapshot, gpus, uid, rank))
        p.daemon = True
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


def time(solver, nccl):
    fprop = []
    bprop = []
    total = caffe.Timer()
    allrd = caffe.Timer()
    for _ in range(len(solver.net.layers)):
        fprop.append(caffe.Timer())
        bprop.append(caffe.Timer())
    display = solver.param.display

    def show_time():
        if solver.iter % display == 0:
            s = '\n'
            for i in range(len(solver.net.layers)):
                s += 'forw %3d %8s ' % (i, solver.net._layer_names[i])
                s += ': %.2f\n' % fprop[i].ms
            for i in range(len(solver.net.layers) - 1, -1, -1):
                s += 'back %3d %8s ' % (i, solver.net._layer_names[i])
                s += ': %.2f\n' % bprop[i].ms
            s += 'solver total: %.2f\n' % total.ms
            s += 'allreduce: %.2f\n' % allrd.ms
            caffe.log(s)
            print s

    solver.net.before_forward(lambda layer: fprop[layer].start())
    solver.net.after_forward(lambda layer: fprop[layer].stop())
    solver.net.before_backward(lambda layer: bprop[layer].start())
    solver.net.after_backward(lambda layer: bprop[layer].stop())
    solver.add_callback(lambda: total.start(), lambda: (total.stop(), allrd.start()))
    solver.add_callback(nccl)
    solver.add_callback(lambda: '', lambda: (allrd.stop(), show_time()))

def plot(solver, nccl):

    display = solver.param.display

    # SET PLOTS DATA
    train_loss = zeros(solver.param.max_iter/display)
    val_loss = zeros(solver.param.max_iter/test_interval)
    it_axes = (arange(solver.param.max_iter) * display) + display
    it_val_axes = (arange(solver.param.max_iter) * test_interval) + test_interval

    _, ax1 = subplots()
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss (r), val loss (g)')
    loss = np.zeros(solver.param.max_iter)


    def do_plot():
        if solver.iter % display == 0:

            loss[solver.iter] = solver.net.blobs['loss3/loss3'].data.copy()
            loss_disp = 'loss=' + str(loss[solver.iter])

            print '%3d) %s' % (solver.iter, loss_disp)

            train_loss[solver.iter / display] = loss[solver.iter]
            ax1.plot(it_axes[0:solver.iter / display], train_loss[0:solver.iter / display], 'r')
            # if it > test_interval:
            #     ax1.plot(it_val_axes[0:it/test_interval], val_loss[0:it/test_interval], 'g') #Val always on top
            ax1.set_ylim([5, 7])
            plt.title(training_id)
            plt.ion()
            plt.grid(True)
            plt.show()
            plt.pause(0.001)

            # VALIDATE Validation done this way only uses 1 GPU
        if solver.iter % test_interval == 0 and solver.iter > 0:
            loss_val = 0
            for i in range(test_iters):
                solver.test_nets[0].forward()
                loss_val += solver.test_nets[0].blobs['loss3/loss3'].data
            loss_val /= test_iters
            print("Val loss: {:.3f}".format(loss_val))

            val_loss[solver.iter / test_interval - 1] = loss_val
            ax1.plot(it_val_axes[0:solver.iter / test_interval], val_loss[0:solver.iter / test_interval], 'g')
            ax1.set_ylim([5, 7])
            plt.title(training_id)
            plt.ion()
            plt.grid(True)
            plt.show()
            plt.pause(0.001)
            title = '../../../datasets/SocialMedia/models/training/' + training_id + str(
                solver.iter) + '.png'  # Save graph to disk
            savefig(title, bbox_inches='tight')

    solver.add_callback(nccl)
    solver.add_callback(lambda: '', lambda: (do_plot()))


def solve(proto, snapshot, gpus, uid, rank):

    print 'Loading solver to GPU: ' + str(rank)

    caffe.set_mode_gpu()
    caffe.set_device(gpus[rank])
    caffe.set_solver_count(len(gpus))
    caffe.set_solver_rank(rank)
    caffe.set_multiprocess(True)

    solver = caffe.SGDSolver(proto)
    if snapshot and len(snapshot) != 0:
        print 'Loading snapshot from : ' + snapshot + '  to GPU: ' + str(rank)
        #solver.restore(snapshot)
        solver.net.copy_from(snapshot)

    nccl = caffe.NCCL(solver, uid)
    nccl.bcast()
    if timing and rank == 0:
        print 'Timing ON'
        time(solver, nccl)
    else:
        solver.add_callback(nccl)

    if plotting and rank == 0:
        print 'Plotting ON'
        plot(solver, nccl)

    if solver.param.layer_wise_reduce:
        solver.net.after_backward(nccl)

    print 'Starting solver for GPU: ' + str(rank)
    solver.step(solver.param.max_iter)

train(solver_path, snapshot, gpus)