import logging
import numpy as np
import sys
sys.path.append('/usr/lib/python2.7/dist-packages')
from scipy.ndimage.interpolation import rotate
sys.path = sys.path[:-1]

logger = logging.getLogger(__name__)

from neon.data import NervanaDataIterator, ArrayIterator

class HDF5Iterator(NervanaDataIterator):

    """
    Iterates over subsamples of an HDF5 volume, returning subimages of a
    certain shape taken from the input and labels datasets, with random
    transforms.
    """

    def __init__(self, inputs, labels, ndata=1024*1024, name=None):
        """
            hdf5_input: input dataset to sample from
            hdf5_labels: labels values to sample from (same sample locations)
            ndata: iterator pretends to have this many examples, for epochs
        """
        super(HDF5Iterator, self).__init__(name=name)
        self.ndata = ndata
        assert self.ndata >= self.be.bsz
        self.start = 0  # how many subimages we have sampled

        # the data to sample from
        self.inputs = inputs
        self.input_lengths = [d.shape[0] for d in inputs]
        self.input_weights = np.array(self.input_lengths, dtype=float) / sum(self.input_lengths)
        self.labels = labels

        # store shape of the input data
        lshape = inputs[0][0, ...].shape
        assert lshape[1] == lshape[2]
        assert len(lshape) == 3
        self.shape = lshape
        self.lshape = lshape
        self.npix = np.prod(lshape)

        # buffers
        self.dev_image = self.be.iobuf(self.npix)
        self.dev_labels = self.be.iobuf(362)
        self.host_image = np.zeros(self.dev_image.shape, dtype=self.dev_image.dtype)
        self.host_labels = np.zeros(self.dev_labels.shape, dtype=self.dev_labels.dtype)

    @property
    def nbatches(self):
        return -((self.start - self.ndata) // self.be.bsz)

    def reset(self):
        """
        For resetting the starting index of this dataset back to zero.
        Relevant for when one wants to call repeated evaluations on the dataset
        but don't want to wrap around for the last uneven minibatch
        Not necessary when ndata is divisible by batch size
        """
        self.start = 0

    def __iter__(self):
        """
        Defines a generator that can be used to iterate over this dataset.
        Yields:
            tuple: The next minibatch which includes both features and labels.
        """
        for minibatch_idx in range(self.start, self.ndata, self.be.bsz):
            for idx in range(self.be.bsz):
                # choose a random move
                rand_group = np.random.choice(len(self.inputs), p=self.input_weights)
                cur_input = self.inputs[rand_group]
                cur_labels = self.labels[rand_group]
                rand_idx = np.random.randint(0, cur_input.shape[0])
                tmp_in = cur_input[rand_idx, ...]
                tmp_la = np.zeros((19, 19))
                dest_row, dest_col = cur_labels[rand_idx, ...]

                if dest_row >= 0:
                    tmp_la[dest_row, dest_col] = 1

                # FLIPS
                if np.random.randint(0, 2) == 1:
                    tmp_in = tmp_in[:, ::-1, :]
                    tmp_la = tmp_la[::-1, :]
                if np.random.randint(0, 2) == 1:
                    tmp_in = tmp_in[:, :, ::-1]
                    tmp_la = tmp_la[:, ::-1]

                # random transpose
                if np.random.randint(0, 2) == 1:
                    tmp_in = tmp_in.transpose((0, 2, 1))
                    tmp_la = tmp_la.transpose((1, 0))

                if dest_row == -1:
                    # PASS
                    out_pos = 361
                else:
                    nr, nc = np.nonzero(tmp_la)
                    out_pos = int(nr * 19 + nc)

                self.host_image[..., idx] = tmp_in.ravel()

                tmp = np.zeros((362,))
                tmp[out_pos] = 1
                self.host_labels[:, idx] = tmp

            self.dev_image[...] = self.host_image
            self.dev_labels[...] = self.host_labels

            yield self.dev_image, self.dev_labels
