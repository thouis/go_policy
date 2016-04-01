import h5py
import logging
import numpy as np
import multiprocessing

logger = logging.getLogger(__name__)

from neon.data import NervanaDataIterator

def generate_sample_stream(filenames, queue, exit_event):
    '''generates individual samples, puts them on a queue to be read elsewhere'''

    hdf5s = [h5py.File(f, 'r') for f in filenames]
    inputs = [h['X'] for h in hdf5s]
    labels = [h['y'] for h in hdf5s]
    input_lengths = [d.shape[0] for d in inputs]
    input_weights = np.array(input_lengths, dtype=float) / sum(input_lengths)

    while not exit_event.is_set():
        # choose a random move
        rand_group = np.random.choice(len(inputs), p=input_weights)
        cur_input = inputs[rand_group]
        cur_labels = labels[rand_group]
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
        tmp_y = np.zeros((362,))
        tmp_y[out_pos] = 1

        queue.put((tmp_in, tmp_y))

class HDF5Iterator(NervanaDataIterator):

    """
    Iterates over subsamples of an HDF5 volume, returning subimages of a
    certain shape taken from the input and labels datasets, with random
    transforms.
    """

    def __init__(self, filenames, inputs, labels, ndata=(1024 * 1024), name=None):
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
        self.filenames = filenames
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

        sample_queue = multiprocessing.Queue(self.be.bsz * 10)
        stop_event = multiprocessing.Event()

        sampler = multiprocessing.Process(target=generate_sample_stream, args=(self.filenames, sample_queue, stop_event))
        sampler.daemon = True
        sampler.start()

        for minibatch_idx in range(self.start, self.ndata, self.be.bsz):
            for idx in range(self.be.bsz):
                tmp_in, tmp_y = sample_queue.get()
                self.host_image[..., idx] = tmp_in.ravel()
                self.host_labels[:, idx] = tmp_y

            self.dev_image[...] = self.host_image
            self.dev_labels[...] = self.host_labels

            yield self.dev_image, self.dev_labels

        stop_event.set()

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
            subset = inputs[sl, ...].transpose((1, 2, 3, 0))
            self.dev_image[...] = subset.reshape((-1, self.be.bsz)).copy().astype(self.dev_image.dtype)

            yield self.dev_image, self.labels[0][sl, ...], sl
