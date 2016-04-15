import sys
import cPickle
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer

from neon.models import Model
from neon.optimizers.optimizer import get_param_list
from neon.backends import gen_backend
import numpy as np

def params(layers):
    for (param, grad), states in get_param_list(layers):
        yield param

def pklify_model():
    layer_list = model.layers.layers_to_optimize
    return cPickle.dumps([(l.name, p.get().astype(np.float16))
                          for l, p in zip(layer_list, params(layer_list))],
                         -1)

def update_model(updates):
    layer_list = model.layers.layers_to_optimize
    for (newl, gradp), (l, p) in zip(updates, zip(layer_list, params(layer_list))):
        assert newl == l
        p[:] = p + gradp

class ModelHandler(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

    def do_GET(self):
        self._set_headers()
        self.wfile.write(pklify_model())

    def do_POST(self):
        self._set_headers()
        data_string = self.rfile.read(int(self.headers['Content-Length']))
        update_model(cPickle.loads(data_string))
        self.wfile.write(pklify_model())


if __name__ == "__main__":
    be = gen_backend(backend='cpu', batch_size=50)
    model = Model(sys.argv[1])

    port = int(sys.argv[2])
    server_address = ('', port)
    httpd = HTTPServer(server_address, ModelHandler)
    print 'Starting httpd...'
    httpd.serve_forever()
