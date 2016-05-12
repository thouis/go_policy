import os.path

from neon.initializers import GlorotUniform, Constant, Uniform
from neon.layers import Conv, GeneralizedCost, Dropout, SkipNode, Activation, Bias, Affine, MergeSum, BatchNorm
from neon.models import Model
from neon.optimizers import GradientDescentMomentum, Schedule, Adadelta
from neon.transforms import Rectlin, CrossEntropyBinary, Accuracy, Softmax, TopKMisclassification
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser
from stochastic import DropAll

from data_iterator import HDF5Iterator

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument("hdf5_list")
parser.add_argument("workspace_dir")
parser.add_argument("--load_model", default=None)
args = parser.parse_args()

# hyperparameters
num_epochs = args.epochs
network_depth = 16
num_features = 128

def conv_params(fsize, relu=True, batch_norm=True):
    padding = {'pad_h': fsize[0] // 2, 'pad_w': fsize[1] // 2}  # always pad to preserve width and height
    return dict(fshape=fsize,
                activation=(Rectlin() if relu else None),
                padding=padding,
                batch_norm=batch_norm,
                init=GlorotUniform())

def resnet_module(nfm, keep_prob=1.0):
    sidepath = SkipNode()
    mainpath = [BatchNorm(),
                Activation(Rectlin()),
                Conv(**conv_params((3, 3, nfm))),
                Dropout(0.9),
                Conv(**conv_params((3, 3, nfm), relu=False, batch_norm=False))]
    return [MergeSum([sidepath, mainpath])]

def build_model(depth, nfm):
    # TODO - per-location bias at each layer

    # input - expand to #nfm feature maps
    layers = [Conv(**conv_params((5, 5, nfm), relu=False, batch_norm=False))]

    for d in range(depth):
        # stochastic depth with falloff from 1.0 to 0.5 from input to final
        # output
        layers += resnet_module(nfm, 1.0 - (0.5 * d) / (depth - 1))

    # reduce to 1 feature map, then affine to 362 outputs
    layers += [Conv(**conv_params((1, 1, 4), relu=False, batch_norm=False)),
               Affine(362,
                      bias=Constant(),
                      init=Uniform(-1.0 / (362 * nfm), 1.0 / (362 * nfm)),
                      activation=Rectlin()),
               Affine(362,
                      bias=Constant(),
                      init=Uniform(-1.0 / (362), 1.0 / (362)),
                      activation=Softmax())]

    return Model(layers=layers)

if args.load_model is None:
    model = build_model(network_depth, num_features)
else:
    print("Starting from {}".format(args.load_model))
    model = Model(args.load_model)

filenames = [s.strip() for s in open(args.hdf5_list)]
train = HDF5Iterator(filenames,
                     ndata=(1024 * 1024),
                     validation=False,
                     remove_history=False)
valid = HDF5Iterator(filenames,
                     ndata=1024,
                     validation=True,
                     remove_history=False)

cost = GeneralizedCost(costfunc=CrossEntropyBinary())

schedule = Schedule(step_config=[2, 10, 20], change=[0.002, 0.001, 0.0001])
opt_adad = Adadelta(decay=0.99, epsilon=1e-6)
opt_gdm = GradientDescentMomentum(learning_rate=0.001,
                                  momentum_coef=0.95,
                                  stochastic_round=args.rounding,
                                  schedule=schedule)

callbacks = Callbacks(model, eval_set=valid, metric=TopKMisclassification(5), **args.callback_args)
callbacks.add_save_best_state_callback(os.path.join(args.workspace_dir, "best_state_h5resnet.pkl"))
#model.fit(train, optimizer=opt_adad, num_epochs=num_epochs, cost=cost, callbacks=callbacks)
model.fit(train, optimizer=opt_gdm, num_epochs=num_epochs, cost=cost, callbacks=callbacks)
model.save_params(os.path.join(args.workspace_dir, "final_state_h5resnet.pkl"))
