import os.path

from neon.initializers import GlorotUniform, Constant, Uniform
from neon.layers import Conv, GeneralizedCost, Dropout, SkipNode, Activation, Affine, MergeSum, BatchNorm
from neon.models import Model
from neon.optimizers import GradientDescentMomentum, ExpSchedule, Adadelta
from neon.transforms import Explin, CrossEntropyMulti, Softmax, TopKMisclassification
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser

from data_iterator import HDF5Iterator

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument("hdf5_list")
parser.add_argument("workspace_dir")
parser.add_argument("--load_model", default=None)
args = parser.parse_args()

# hyperparameters
num_epochs = args.epochs
network_depth = 20
num_features = 256


def conv_params(fsize, elu=True, batch_norm=True):
    padding = {'pad_h': fsize[0] // 2, 'pad_w': fsize[1] // 2}  # always pad to preserve width and height
    return dict(fshape=fsize,
                activation=(Explin() if elu else None),
                padding=padding,
                batch_norm=batch_norm,
                init=GlorotUniform())


def resnet_module(nfm, keep_prob=1.0):
    print("layer with droupout keep = {}".format(keep_prob))
    sidepath = [SkipNode(), Dropout(keep_prob)]
    mainpath = [BatchNorm(),
                Activation(Explin()),
                Conv(**conv_params((3, 3, nfm))),
                Conv(**conv_params((3, 3, nfm), elu=False, batch_norm=False)),
                Dropout(keep_prob)]
    return [MergeSum([sidepath, mainpath])]


def build_model(depth, nfm):
    # input - expand to #nfm feature maps
    layers = [Conv(**conv_params((3, 3, nfm), elu=False, batch_norm=False))]

    for d in range(depth):
        # Swapout with falloff from 1.0 to 0.5 from input to final
        # output
        layers += resnet_module(nfm, 1.0 - (0.5 * d) / (depth - 1))

    # reduce to 4 feature maps (minimum for neon), then affine to 362 outputs
    layers += [Conv(**conv_params((1, 1, 4), elu=False, batch_norm=False)),
               Affine(362,
                      bias=Constant(),
                      init=Uniform(-1.0 / (362 * nfm), 1.0 / (362 * nfm)),
                      activation=Explin()),
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
                     remove_history=False,
                     minimal_set=True)
valid = HDF5Iterator(filenames,
                     ndata=(16 * 1024),
                     validation=True,
                     remove_history=False,
                     minimal_set=True)

cost = GeneralizedCost(costfunc=CrossEntropyMulti(usebits=True))

schedule = ExpSchedule(decay=(1.0 / 50))  # halve the learning rate every 50 epochs
opt_gdm = GradientDescentMomentum(learning_rate=0.01,
                                  momentum_coef=0.9,
                                  stochastic_round=args.rounding,
                                  gradient_clip_norm=5,
                                  schedule=schedule)

callbacks = Callbacks(model, eval_set=valid, metric=TopKMisclassification(5), **args.callback_args)
callbacks.add_save_best_state_callback(os.path.join(args.workspace_dir, "best_state_h5resnet.pkl"))
model.fit(train, optimizer=opt_gdm, num_epochs=num_epochs, cost=cost, callbacks=callbacks)
model.save_params(os.path.join(args.workspace_dir, "final_state_h5resnet.pkl"))
