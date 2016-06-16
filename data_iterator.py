import h5py
import logging
import numpy as np
import multiprocessing
import sys
from neon.data import NervanaDataIterator

logger = logging.getLogger(__name__)


def generate_sample_stream(filenames, queue, validation=False, remove_history=False, minimal_set=False):
    '''generates individual samples, puts them on a queue to be read elsewhere'''

    hdf5s = [h5py.File(f, 'r') for f in filenames]
    inputs = [h['X'] for h in hdf5s]
    labels = [h['y'] for h in hdf5s]
    input_lengths = [d.shape[0] for d in inputs]
    input_weights = np.array(input_lengths, dtype=float) / sum(input_lengths)

    np.random.seed()
    while True:
        # choose a random move.  If from validation, only moves from the first 500 in the data are chosen
        if validation:
            rand_group = np.random.choice(len(inputs))
        else:
            rand_group = np.random.choice(len(inputs), p=input_weights)

        cur_input = inputs[rand_group]
        cur_labels = labels[rand_group]
        validation_base = min(500, cur_input.shape[0] // 2)

        if validation:
            rand_idx = np.random.choice(validation_base)
        else:
            rand_idx = np.random.randint(validation_base, cur_input.shape[0])

        try:
            tmp_in = cur_input[rand_idx, ...]
        except Exception:
            print("die", rand_idx, filenames[rand_group], rand_idx, sys.exc_info())
            raise

        assert tmp_in.shape[-1] == tmp_in.shape[-2] == 19

        if remove_history:  # remove most recent move indicators
            tmp_in[4:12, :, :] = 0
        if minimal_set:
            # player, opponent, empty, legal
            tmp_in = tmp_in[(0, 1, 2, -1), :, :]

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
            assert len(nr) == 1
            assert len(nc) == 1
            out_pos = int(nr * 19 + nc)
        tmp_y = np.zeros((362,))
        tmp_y[out_pos] = 1

        queue.put((tmp_in, tmp_y))


class HDF5Iterator(NervanaDataIterator):
    """
    Iterates over subsamples of an HDF5 volume, returning subimages of a
    certain shape taken from the input and labels datasets, with random
    transforms.
    """

    def __init__(self, filenames, ndata=(1024 * 1024), name=None,
                 validation=False, remove_history=False, minimal_set=False):
        """
            hdf5_input: input dataset to sample from
            hdf5_labels: labels values to sample from (same sample locations)
            ndata: iterator pretends to have this many examples, for epochs
        """
        super(HDF5Iterator, self).__init__(name=name)
        self.ndata = ndata
        assert self.ndata >= self.be.bsz
        self.start = 0  # how many subimages we have sampled
        self.validation = validation
        self.remove_history = remove_history
        self.minimal_set = minimal_set

        # the data to sample from
        self.filenames = filenames

        # HDF5 sampler subprocess
        self.sample_queue = multiprocessing.Queue(self.be.bsz * 10)
        for idx in range(10):
            self.sampler = multiprocessing.Process(target=generate_sample_stream,
                                                   args=(self.filenames, self.sample_queue,
                                                         self.validation, self.remove_history,
                                                         self.minimal_set))

            self.sampler.daemon = True
            self.sampler.start()

        # store shape of the input data
        first_batch = self.sample_queue.get()
        lshape = first_batch[0].shape
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
                tmp_in, tmp_y = self.sample_queue.get()
                self.host_image[..., idx] = tmp_in.ravel()
                self.host_labels[:, idx] = tmp_y

            self.dev_image[...] = self.host_image
            self.dev_labels[...] = self.host_labels

            yield self.dev_image, self.dev_labels

    def predict(self):
        """
        Defines a generator that can be used to iterate over this dataset in order
        Yields:
            tuple: The next batch which includes data and slice
        """
        assert len(self.inputs) == 1
        inputs = self.inputs[0]

        for offset in range(self.start, inputs.shape[0], self.be.bsz):
            offset = min(offset, inputs.shape[0] - self.be.bsz)
            sl = slice(offset, offset + self.be.bsz)
            # neon wants CHWN
            subset = inputs[sl, ...]
            if self.remove_history:
                subset[:, 4:12, :, :] = 0
            subset = subset.transpose((1, 2, 3, 0))
            self.dev_image[...] = subset.reshape((-1, self.be.bsz)).copy().astype(self.dev_image.dtype)

            yield self.dev_image, self.labels[0][sl, ...], sl
