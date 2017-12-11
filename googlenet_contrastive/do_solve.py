def do_solve(niter, solvers, disp_interval, test_interval, test_iters, training_id, batch_size):
    """Run solvers for niter iterations,
       returning the loss and recorded each iteration.
       `solvers` is a list of (name, solver) tuples."""

    import tempfile
    import numpy as np
    import os
    from pylab import zeros, arange, subplots, plt, savefig
    import time

    # SET PLOTS DATA
    train_loss = zeros(niter/disp_interval)
    train_correct_pairs = zeros(niter/disp_interval)

    val_loss = zeros(niter/test_interval)
    val_correct_pairs = zeros(niter/test_interval)


    it_axes = (arange(niter) * disp_interval) + disp_interval
    it_val_axes = (arange(niter) * test_interval) + test_interval

    _, ax1 = subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss (r), val loss (g)')
    ax2.set_ylabel('train correct pairs (b) val correct pairs (m)')
    ax2.set_autoscaley_on(False)
    ax2.set_ylim([0, batch_size])


    loss = {name: np.zeros(niter) for name, _ in solvers}
    correct_pairs = {name: np.zeros(niter) for name, _ in solvers}


    #RUN TRAINING
    for it in range(niter):
        for name, s in solvers:
            # start = time.time()
            s.step(1)  # run a single SGD step in Caffe
            # end = time.time()
            # print "Time step: " + str((end - start))
            # print "Max before ReLU: " + str(np.max(s.net.blobs['inception_5b/pool_proj'].data))
            # print "Max last FC: " + str(np.max(s.net.blobs['loss3/classifierCustom'].data))

            loss[name][it] = s.net.blobs['loss3/loss3'].data.copy()
            correct_pairs[name][it] = s.net.blobs['correct_pairs'].data.copy()

        #PLOT
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = 'loss=' + str(loss['my_solver'][it]) + ' correct_pairs=' + str(correct_pairs['my_solver'][it])

            print '%3d) %s' % (it, loss_disp)

            train_loss[it/disp_interval] = loss['my_solver'][it]
            train_correct_pairs[it/disp_interval] = correct_pairs['my_solver'][it]


            ax1.plot(it_axes[0:it/disp_interval], train_loss[0:it/disp_interval], 'r')
            ax2.plot(it_axes[0:it/disp_interval], train_correct_pairs[0:it/disp_interval], 'b')

            # if it > test_interval:
            #     ax1.plot(it_val_axes[0:it/test_interval], val_loss[0:it/test_interval], 'g') #Val always on top
            ax1.set_ylim([0,5])
            plt.title(training_id)
            plt.ion()
            plt.grid(True)
            plt.show()
            plt.pause(0.001)
            # title = '../training/numbers/training-' + str(it) + '.png'  # Save graph to disk
            # savefig(title, bbox_inches='tight')

        #VALIDATE
        if it % test_interval == 0 and it > 0:
            loss_val = 0
            cur_correct_pairs = 0
            for i in range(test_iters):
                solvers[0][1].test_nets[0].forward()
                loss_val += solvers[0][1].test_nets[0].blobs['loss3/loss3'].data
                cur_correct_pairs += solvers[0][1].test_nets[0].blobs['correct_pairs'].data

            loss_val /= test_iters
            cur_correct_pairs /= test_iters

            print("Val loss: " + str(loss_val) + " Val correct pairs: " + str(cur_correct_pairs))

            val_loss[it/test_interval - 1] = loss_val
            val_correct_pairs[it/test_interval - 1] = cur_correct_pairs

            ax1.plot(it_val_axes[0:it/test_interval], val_loss[0:it/test_interval], 'g')
            ax2.plot(it_val_axes[0:it/test_interval], val_correct_pairs[0:it/test_interval], 'm')
            ax1.set_ylim([0,5])
            plt.title(training_id)
            plt.ion()
            plt.grid(True)
            plt.show()
            plt.pause(0.001)
            title = '../../../datasets/SocialMedia/models/training/' + training_id + str(it) + '.png'  # Save graph to disk
            savefig(title, bbox_inches='tight')


    #Save the learned weights from both nets at the end of the training
    weight_dir = tempfile.mkdtemp()
    weights = {}
    for name, s in solvers:
        filename = 'weights.%s.caffemodel' % name
        weights[name] = os.path.join(weight_dir, filename)
        s.net.save(weights[name])

    return loss, weights
