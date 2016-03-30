import h5py
import os.path

from neon.initializers import GlorotUniform, Constant, Uniform
from neon.layers import Conv, GeneralizedCost, Dropout, MergeSum, SkipNode, Activation, Bias, Affine
from neon.models import Model
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.transforms import Rectlin, CrossEntropyBinary, Accuracy, Softmax
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser

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

def resnet_module(nfm):
    sidepath = SkipNode()
    mainpath = [Conv(**conv_params((3, 3, nfm))),
                Bias(Constant()),
                Conv(**conv_params((3, 3, nfm), relu=False)),
                Bias(Constant())]
    return [MergeSum([mainpath, sidepath]),
            Activation(Rectlin())]

def build_model(depth, nfm):
    # TODO - per-location bias at each layer

    # input - expand to #nfm feature maps
    layers = [Conv(**conv_params((3, 3, nfm))), Dropout(0.8)]

    for d in range(depth):
        layers += resnet_module(nfm)

    # final output: 1 channel
    layers += [Dropout(),
               Affine(362, init=Uniform(-1.0 / (362 * nfm), 1.0 / (362 * nfm)), activation=Softmax())]


    return Model(layers=layers)

model = build_model(network_depth, num_features)

h5s = [h5py.File(s.strip()) for s in open(args.hdf5_list)]
num_moves = sum(h['X'].shape[0] for h in h5s)
print("Found {} HDF5 files with {} moves".format(len(h5s), num_moves))
train = HDF5Iterator([h['X'] for h in h5s],
                     [h['y'] for h in h5s],
                     ndata=1024*1024)
valid = HDF5Iterator([h['X'] for h in h5s],
                     [h['y'] for h in h5s],
                     ndata=1024)

cost = GeneralizedCost(costfunc=CrossEntropyBinary())

schedule = Schedule(step_config=[5, 10], change=[0.0001, 0.00001])
opt_gdm = GradientDescentMomentum(learning_rate=0.001,
                                  momentum_coef=0.9,
                                  stochastic_round=args.rounding,
                                  schedule=schedule)

callbacks = Callbacks(model, eval_set=valid, metric=Accuracy(), **args.callback_args)
callbacks.add_save_best_state_callback(os.path.join(args.workspace_dir, "best_state_h5resnet.pkl"))
model.fit(train, optimizer=opt_gdm, num_epochs=num_epochs, cost=cost, callbacks=callbacks)
model.save_params(os.path.join(args.workspace_dir, "final_state_h5resnet.pkl"))
