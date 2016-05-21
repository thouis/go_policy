import h5py
import numpy as np
import multiprocessing
import sys


def generate_sample_stream(filenames, queue, validation=False, remove_history=False):
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

class HDF5Iterator(object):

    """
    Iterates over subsamples of an HDF5 volume, returning subimages of a
    certain shape taken from the input and labels datasets, with random
    transforms.
    """

    def __init__(self, filenames, ndata=(1024 * 1024), batch_size=64,
                 name=None, validation=False, remove_history=False):
        """
            hdf5_input: input dataset to sample from
            ndata: iterator pretends to have this many examples, for epochs
        """
        super(HDF5Iterator, self).__init__(name=name)
        self.ndata = ndata
        self.batch_size = batch_size
        self.validation = validation
        self.remove_history = remove_history

        # the data to sample from
        self.filenames = filenames

        # HDF5 sampler subprocess
        self.sample_queue = multiprocessing.Queue(self.be.bsz * 10)
        for idx in range(10):
            self.sampler = multiprocessing.Process(target=generate_sample_stream, args=(self.filenames, self.sample_queue, self.validation, remove_history))
            self.sampler.daemon = True
            self.sampler.start()

        # store shape of the input data
        first_batch = self.sample_queue.get()
        lshape = first_batch[0].shape
        assert lshape[1] == lshape[2]
        assert len(lshape) == 3
        self.lshape = lshape

    def new_buffers(self):
        dest_input = np.empty((self.batch_size,) + self.lshape, dtype=np.uint8)
        dest_labels = np.empty([self.batch_size, 362], dtype=np.uint8)
        return dest_input, dest_labels

    def __iter__(self):
        """
        Defines a generator that can be used to iterate over this dataset.
        Yields:
            tuple: The next minibatch which includes both features and labels.
        """
        for minibatch_idx in range(self.ndata):
            buf_input, buf_labels = self.new_buffers()
            for idx in range(self.batch_size):
                tmp_in, tmp_y = self.sample_queue.get()
                buf_input[idx, ...] = tmp_in
                buf_labels[idx, ...] = tmp_y

            yield buf_input, buf_labels
