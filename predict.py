import h5py
import os.path
import numpy as np

from neon.models import Model
from neon.util.argparser import NeonArgparser

from data_iterator import HDF5Iterator

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument("hdf5")
parser.add_argument("model_pkl")
args = parser.parse_args()

model = Model(args.model_pkl)

h5s = [h5py.File(args.hdf5)]
num_moves = sum(h['X'].shape[0] for h in h5s)
print("Found {} HDF5 files with {} moves".format(len(h5s), num_moves))
inputs = HDF5Iterator([h['X'] for h in h5s],
                      [h['y'] for h in h5s],
                      ndata=(1024 * 1024))

out_predict = h5s[0].require_dataset("predictions", (num_moves, 362), dtype=np.float32)
out_score = h5s[0].require_dataset("scores", (num_moves,), dtype=np.float32)
out_max = h5s[0].require_dataset("best", (num_moves,), dtype=np.float32)


model.initialize(inputs)
for indata, actual, sl in inputs.predict():
    prediction = model.fprop(indata, inference=False).get().T
    actual = actual.astype(int)
    actual_idx = actual[:, 0] * 19 + actual[:, 1]
    actual_idx[actual_idx < 0] = 361
    out_predict[sl, :] = prediction
    out_score[sl] = prediction[range(prediction.shape[0]), actual_idx]
    out_max[sl] = prediction.max(axis=1)
    print (sl)
