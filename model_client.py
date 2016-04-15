import os.path
import cPickle
import h5py
import requests

from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.optimizers.optimizer import get_param_list
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser
from neon.layers import GeneralizedCost
from neon.transforms import CrossEntropyBinary, TopKMisclassification

from data_iterator import HDF5Iterator

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument("hdf5_list")
parser.add_argument("workspace_dir")
parser.add_argument("model_pkl")
parser.add_argument("server_address")
args = parser.parse_args()

def get_model_params(url):
    r = requests.get(url)
    return cPickle.loads(r.text)

def put_deltas(url, deltas):
    r = requests.post(url, data=cPickle.dumps(deltas))
    return cPickle.loads(r.text)

def params(layers):
    for (param, grad), states in get_param_list(layers):
        yield param

def update_model(model, newparams):
    layer_list = model.layers.layers_to_optimize
    for (newl, newp), (l, p) in zip(newparams, zip(layer_list, params(layer_list))):
        assert newl == l.name
        p[:] = newp

def compute_deltas(oldparams, model):
    deltas = []
    layer_list = model.layers.layers_to_optimize
    for (oldl, oldp), (l, p) in zip(oldparams, zip(layer_list, params(layer_list))):
        assert oldl == l.name
        deltas.append((l.name, p[:].get() - oldp))
    return deltas

# hyperparameters
num_epochs = 1
print("Starting from {}".format(args.model_pkl))
model = Model(args.model_pkl)

filenames = [s.strip() for s in open(args.hdf5_list)]
h5s = [h5py.File(f) for f in filenames]
num_moves = sum(h['X'].shape[0] for h in h5s)
print("Found {} HDF5 files with {} moves".format(len(h5s), num_moves))
train = HDF5Iterator(filenames,
                     [h['X'] for h in h5s],
                     [h['y'] for h in h5s],
                     ndata=(256 * 1024),
                     validation=False,
                     remove_history=True)
valid = HDF5Iterator(filenames,
                     [h['X'] for h in h5s],
                     [h['y'] for h in h5s],
                     ndata=1024,
                     validation=True,
                     remove_history=True)

cost = GeneralizedCost(costfunc=CrossEntropyBinary())
opt_gdm = GradientDescentMomentum(learning_rate=0.01,
                                  momentum_coef=0.9,
                                  stochastic_round=args.rounding)

callbacks = Callbacks(model, eval_set=valid, metric=TopKMisclassification(5), **args.callback_args)

old_params = get_model_params(args.server_address)
num_iterations = 1
while True:
    update_model(model, old_params)
    model.fit(train, optimizer=opt_gdm, num_epochs=1, cost=cost, callbacks=callbacks)
    model.save_params(os.path.join(args.workspace_dir, "iter_{}.pkl".format(num_iterations)))

    deltas = compute_deltas(old_params, model)
    old_params = put_deltas(args.server_address, deltas)
    num_iterations += 1
