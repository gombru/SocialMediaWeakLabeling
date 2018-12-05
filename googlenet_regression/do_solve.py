def do_solve(niter, solver, disp_interval, test_interval, test_iters, training_id):
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
    val_loss = zeros(niter/test_interval)

    it_axes = (arange(niter) * disp_interval) + disp_interval
    it_val_axes = (arange(niter) * test_interval) + test_interval

    _, ax1 = subplots()
    # ax2 = ax1.twinx()
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss (r), val loss (g)')
    # ax2.set_ylabel('val loss (g)')
    # ax2.set_autoscaley_on(False)
    # ax2.set_ylim([0, 1])

    lowest_val_loss = 1000
    best_it = 0
    loss = np.zeros(niter)


    #RUN TRAINING
    for it in range(niter):
        # start = time.time()
        solver.step(1)  # run a single SGD step in Caffe
        # end = time.time()
        # print "Time step: " + str((end - start))
        loss[it] = solver.net.blobs['loss3/loss3'].data.copy()

        #PLOT
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = 'loss=' + str(loss[it])

            print '%3d) %s' % (it, loss_disp)

            train_loss[it/disp_interval] = loss[it]

            ax1.plot(it_axes[0:it/disp_interval], train_loss[0:it/disp_interval], 'r')
            # if it > test_interval:
            #     ax1.plot(it_val_axes[0:it/test_interval], val_loss[0:it/test_interval], 'g') #Val always on top
            ax1.set_ylim([0,300])
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
            for i in range(test_iters):
                solver.test_nets[0].forward()
                loss_val += solver.test_nets[0].blobs['loss3/loss3'].data
            loss_val /= test_iters
            print("Val loss: {:.3f}".format(loss_val))

            val_loss[it/test_interval - 1] = loss_val
            ax1.plot(it_val_axes[0:it/test_interval], val_loss[0:it/test_interval], 'g')
            ax1.set_ylim([0,300])
            ax1.set_xlabel('iteration ' + 'Best it: ' + str(best_it) + ' Best Val Loss: ' + str(int(lowest_val_loss))
            plt.title(training_id)
            plt.ion()
            plt.grid(True)
            plt.show()
            plt.pause(0.001)
            title = '../../../hd/datasets/instaFashion/models/training/' + training_id + str(it) + '.png'  # Save graph to disk
            savefig(title, bbox_inches='tight')

            if loss_val < lowest_val_loss:
                print("Best Val loss!")
                lowest_val_loss = loss_val
                best_it = it
                filename = '../../../hd/datasets/instaFashion/models/CNNRegression/' + training_id + 'best_valLoss_' + str(int(loss_val)) +'_it_' + str(it) + '.caffemodel'
                prefix = 30
                for cur_filename in glob.glob(filename[:-prefix] + '*'):
                    print(cur_filename)
                    os.remove(cur_filename)
                solver.net.save(filename)