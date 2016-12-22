import h5py
import logging
import numpy as np
import multiprocessing
from neon.data import NervanaDataIterator

logger = logging.getLogger(__name__)


def do_flips(arr, flip_horiz, flip_vert, transpose):
    if flip_horiz:
        arr = arr[:, :, ::-1]
    if flip_vert:
        arr = arr[:, ::-1, :]
    if transpose:
        arr = arr.transpose((0, 2, 1))
    return arr


def generate_sample_stream(filenames, queue, validation=False, remove_history=False, minimal_set=False, next_N=1):
    '''generates individual samples, puts them on a queue to be read elsewhere'''

    np.seterr(all='raise')

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
            rand_idx = np.random.randint(validation_base, cur_input.shape[0] - next_N + 1)

        actions = cur_labels[rand_idx:(rand_idx + next_N), ...].copy()
        # if there's a pass, assume the other side passed as well, and replace all the moves with pass
        # TODO - maybe we need an "end of game" marker in the processed SGFs
        saw_pass = False
        for offset in range(next_N):
            saw_pass = saw_pass or actions[offset, 0] == -1
            if saw_pass:
                actions[offset, :] = [-1, -1]

        state = cur_input[rand_idx, ...].copy()
        assert state.shape[-1] == state.shape[-2] == 19

        if remove_history:  # remove most recent move indicators
            state[4:12, :, :] = 0
        if minimal_set:
            # player, opponent, empty, legal
            state = state[(0, 1, 2, -1), :, :]

        flip1 = np.random.randint(0, 2)
        flip2 = np.random.randint(0, 2)
        transpose = np.random.randint(0, 2)

        state = do_flips(state, flip1, flip2, transpose)

        actions_linear = np.zeros((next_N, 362))
        for offset in range(next_N):
            if actions[offset, 0] != -1:
                action_pos = np.zeros((1, 19, 19))
                action_pos[0,
                           actions[offset, 0],
                           actions[offset, 1]] = 1
                action_pos = do_flips(action_pos,
                                      flip1, flip2, transpose)
                _, nr, nc = np.nonzero(action_pos)
                actions_linear[offset, nr * 19 + nc] = 1
            else:
                actions_linear[offset, 361] = 1

        queue.put((state, actions_linear))


class HDF5Iterator(NervanaDataIterator):
    """
    Iterates over subsamples of an HDF5 volume, returning subimages of a
    certain shape taken from the input and labels datasets, with random
    transforms.
    """

    def __init__(self, filenames, ndata=(1024 * 1024), name=None,
                 validation=False, remove_history=False, minimal_set=False, next_N=1):
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
        self.next_N = next_N

        # the data to sample from
        self.filenames = filenames

        # HDF5 sampler subprocess
        self.sample_queue = multiprocessing.Queue(self.be.bsz * 10)
        for idx in range(10):
            self.sampler = multiprocessing.Process(target=generate_sample_stream,
                                                   args=(self.filenames, self.sample_queue,
                                                         self.validation, self.remove_history,
                                                         self.minimal_set, self.next_N))

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
        self.dev_labels = [self.be.iobuf(362) for idx in range(self.next_N)]
        self.host_image = np.zeros(self.dev_image.shape, dtype=self.dev_image.dtype)
        self.host_labels = [np.zeros(self.dev_labels[0].shape, dtype=self.dev_labels[0].dtype) for idx in range(self.next_N)]

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
                state, tmp_y = self.sample_queue.get()
                self.host_image[..., idx] = state.ravel()
                for r in range(self.next_N):
                    self.host_labels[r][:, idx] = tmp_y[r, :]

            self.dev_image[...] = self.host_image
            for r in range(self.next_N):
                self.dev_labels[r][...] = self.host_labels[r]

            if self.next_N == 1:
                yield self.dev_image, self.dev_labels[0]
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
