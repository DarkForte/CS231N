import numpy as np
from cs231n.classifiers.convnet import *
from cs231n.gradient_check import eval_numerical_gradient

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

num_inputs = 2
input_dim = (3, 16, 16)
reg = 0.0
num_classes = 10
X = np.random.randn(num_inputs, *input_dim)
y = np.random.randint(num_classes, size=num_inputs)

model = ConvNet(num_filters=[], filter_size=[],
                input_dim=input_dim, hidden_dim=[4, 4],
                dtype=np.float64)
loss, grads = model.loss(X, y)
for param_name in sorted(grads):
  f = lambda _: model.loss(X, y)[0]
  param_grad_num = eval_numerical_gradient(f, model.params[param_name], verbose=False, h=1e-6)


  print param_grad_num
  print grads[param_name]

  e = rel_error(param_grad_num, grads[param_name])
  print '%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name]))
