def do_solve(niter, solvers, disp_interval, test_interval, test_iters):
    """Run solvers for niter iterations,
       returning the loss and accuracy recorded each iteration.
       `solvers` is a list of (name, solver) tuples."""

    import tempfile
    import numpy as np
    import os
    from pylab import zeros, arange, subplots, plt, savefig

    # SET PLOTS DATA
    train_loss = zeros(niter/disp_interval)
    train_acc = zeros(niter/disp_interval)
    val_acc = zeros(niter/test_interval)

    it_axes = (arange(niter) * disp_interval) + disp_interval
    it_val_axes = (arange(niter) * test_interval) + test_interval

    _, ax1 = subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss (r)')
    ax2.set_ylabel('train accuracy (b), val accuracy (g)')
    ax2.set_autoscaley_on(False)
    ax2.set_ylim([0, 1])

    blobs = ('loss', 'acc')
    loss, acc = ({name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs)


    #RUN TRAINING
    for it in range(niter):
        for name, s in solvers:
            s.step(1)  # run a single SGD step in Caffe
            loss[name][it], acc[name][it] = (s.net.blobs[b].data.copy()
                                             for b in blobs)


        #PLOT
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = '; '.join('%s: loss=%.3f, acc=%2d%%' %
                                  (n, loss[n][it], np.round(100*acc[n][it]))
                                  for n, _ in solvers)
            print '%3d) %s' % (it, loss_disp)

            train_loss[it/disp_interval] = loss['my_solver'][it]
            train_acc[it/disp_interval] = acc['my_solver'][it]

            ax1.plot(it_axes[0:it/disp_interval], train_loss[0:it/disp_interval], 'r')
            ax2.plot(it_axes[0:it/disp_interval], train_acc[0:it/disp_interval], 'b')
            plt.ion()
            plt.show()
            plt.pause(0.001)
            # title = '../training/numbers/training-' + str(it) + '.png'  # Save graph to disk
            # savefig(title, bbox_inches='tight')

        #VALIDATE
        if it % test_interval == 0 and it > 0:
            accuracy = 0
            for i in range(test_iters):
                solvers[0][1].test_nets[0].forward()
                accuracy += solvers[0][1].test_nets[0].blobs['acc'].data
            accuracy /= test_iters
            print("Test Accuracy: {:.3f}".format(accuracy))

            val_acc[it/test_interval - 1] = accuracy
            ax2.plot(it_val_axes[0:it/test_interval], val_acc[0:it/test_interval], 'g')
            plt.ion()
            plt.show()
            plt.pause(0.001)
            title = 'training/training-' + str(it) + '.png'  # Save graph to disk
            savefig(title, bbox_inches='tight')


    #Save the learned weights from both nets at the end of the training
    weight_dir = tempfile.mkdtemp()
    weights = {}
    for name, s in solvers:
        filename = 'weights.%s.caffemodel' % name
        weights[name] = os.path.join(weight_dir, filename)
        s.net.save(weights[name])

    return loss, acc, weights