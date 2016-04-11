from neon.layers.layer import Layer, interpret_in_shape

class DropAll(Layer):
    """
    A stochastic drop layer (drops its entire input).
    This can be used to create Stochastic Depth Networks.
    http://arxiv.org/abs/1603.09382v1

    (though without the speedup they get in training from not
    evaluating certain layers)

    A single keep value applies to all values in the output.

    Each fprop call generates an new keep mask stochastically where there
    distribution of ones in the mask is controlled by the keep param.

    Arguments:
       keep (float): fraction of time the input should be stochastically kept.
    """

    def __init__(self, keep=0.5, name=None):
        super(DropAll, self).__init__(name)
        self.keep = keep
        self.keep_mask = None
        self.owns_output = False

    def __str__(self):
        return "DropAll Layer '%s': %d inputs and outputs, keep %d%%" % (
               self.name, self.nout, 100 * self.keep)

    def configure(self, in_obj):
        super(DropAll, self).configure(in_obj)
        self.out_shape = self.in_shape
        (self.nout, _) = interpret_in_shape(self.in_shape)
        return self

    def allocate(self, shared_outputs=None):
        super(DropAll, self).allocate(shared_outputs)
        self.keep_mask = self.be.zeros((1,), persist_values=True, parallel=self.parallelism)

    def fprop(self, inputs, inference=False):
        self.outputs = self.inputs = inputs
        if inference:
            return self._fprop_inference(inputs)

        self.be.make_binary_mask(self.keep_mask, self.keep)
        self.outputs[:] = self.keep_mask * inputs * (1.0 / self.keep)

        return self.outputs

    def _fprop_inference(self, inputs):
        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        if not self.deltas:
            self.deltas = error
        self.deltas[:] = self.keep_mask * error * alpha * (1.0 / self.keep) + beta * error
        return self.deltas
