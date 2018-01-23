from caffe import layers as L, params as P
import caffe

def l2normed(dim):
    n = caffe.NetSpec()
    n.data, n.label = L.Python(module='layers', layer='tripletDataLayer',
            ntop=2)
    """Returns L2-normalized instances of vec; i.e., for each instance x in vec,
    computes  x / ((x ** 2).sum() ** 0.5). Assumes vec has shape N x dim."""
    n.denom = L.Reduction(n.data, axis=1, operation=P.Reduction.SUMSQ)
    #denom = L.Power(denom, power=(-0.5))
    n.power = L.Power(n.denom, power=(-0.5), shift=1e-12) # For numerical stability
    n.reshape = L.Reshape(n.power, num_axes=0, axis=-1, shape=dict(dim=[1]))
    n.tile = L.Tile(n.reshape, axis=1, tiles=dim)
    n.elwise = L.Eltwise(n.data, n.tile, operation=P.Eltwise.PROD)
    return n.to_proto()

with open('l2normalization.prototxt', 'w') as f:
    f.write(str(l2normed(0)))
