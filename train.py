import h5py
import os.path

from neon.initializers import GlorotUniform, Constant, Uniform
from neon.layers import Conv, GeneralizedCost, Dropout, SkipNode, Activation, Bias, Affine, MergeSum
from neon.models import Model
from neon.optimizers import GradientDescentMomentum, Schedule, Adadelta
from neon.transforms import Rectlin, CrossEntropyBinary, Accuracy, Softmax
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser
from stochastic import DropAll

from data_iterator import HDF5Iterator

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument("hdf5_list")
parser.add_argument("workspace_dir")
args = parser.parse_args()

# hyperparameters
num_epochs = args.epochs
network_depth = 16
num_features = 128

def conv_params(fsize, relu=True, batch_norm=True):
    padding = {'pad_h': 1, 'pad_w': 1}  # always pad to preserve width and height
    return dict(fshape=fsize,
                activation=(Rectlin() if relu else None),
                padding=padding,
                batch_norm=batch_norm,
                init=GlorotUniform())

def resnet_module(nfm, keep_prob=1.0):
    sidepath = SkipNode()
    mainpath = [Conv(**conv_params((3, 3, nfm))),
                Bias(Constant()),
                Conv(**conv_params((3, 3, nfm), relu=False)),
                Bias(Constant()),
                DropAll(keep_prob)]
    return [MergeSum([mainpath, sidepath]),
            Activation(Rectlin())]

def build_model(depth, nfm):
    # TODO - per-location bias at each layer

    # input - expand to #nfm feature maps
    layers = [Conv(**conv_params((3, 3, nfm))), Dropout(0.8)]

    for d in range(depth):
        # stochastic depth with falloff from 1.0 to 0.5 from input to final
        # output
        layers += resnet_module(nfm, 1.0 - (0.5 * d) / (depth - 1))

    # final output: 1 channel
    layers += [Dropout(),
               Affine(362, init=Uniform(-1.0 / (362 * nfm), 1.0 / (362 * nfm)), activation=Softmax())]

    return Model(layers=layers)

model = build_model(network_depth, num_features)

filenames = [s.strip() for s in open(args.hdf5_list)]
h5s = [h5py.File(f) for f in filenames]
num_moves = sum(h['X'].shape[0] for h in h5s)
print("Found {} HDF5 files with {} moves".format(len(h5s), num_moves))
train = HDF5Iterator(filenames,
                     [h['X'] for h in h5s],
                     [h['y'] for h in h5s],
                     ndata=(1024 * 1024))
valid = HDF5Iterator(filenames,
                     [h['X'] for h in h5s],
                     [h['y'] for h in h5s],
                     ndata=1024)

cost = GeneralizedCost(costfunc=CrossEntropyBinary())

schedule = Schedule(step_config=[10, 20], change=[0.001, 0.0001])
opt_adad = Adadelta(decay=0.99, epsilon=1e-6)
opt_gdm = GradientDescentMomentum(learning_rate=0.01,
                                  momentum_coef=0.9,
                                  stochastic_round=args.rounding,
                                  schedule=schedule)

callbacks = Callbacks(model, eval_set=valid, metric=Accuracy(), **args.callback_args)
callbacks.add_save_best_state_callback(os.path.join(args.workspace_dir, "best_state_h5resnet.pkl"))
model.fit(train, optimizer=opt_adad, num_epochs=num_epochs, cost=cost, callbacks=callbacks)
model.save_params(os.path.join(args.workspace_dir, "final_state_h5resnet.pkl"))
