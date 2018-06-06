import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import math_ops

@ops.RegisterGradient("BinaryOut")
def _binary_out_grad(op, grad):
  """The gradients for `binary_out`.

  Args:
    op: The `binary_out` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `binary_out` op.

  Returns:
    Gradients with respect to the input of `binary_out`.
  """
  to_binary = op.inputs[0]
  to_binary = math_ops.abs(to_binary)
  one = array_ops.ones_like(to_binary)
  zero = array_ops.zeros_like(to_binary)
  index = tf.less_equal(to_binary, one)
  to_binary_grad  = tf.where(index, one, zero)
  #shape = array_ops.shape(to_binary)
  #index = array_ops.zeros_like(shape)
  #first_grad = array_ops.reshape(grad, [-1])[0]
  #to_zero_grad = sparse_ops.sparse_to_dense([index], shape, first_grad, 0)
  to_binary_grad = grad * to_binary_grad
  return [to_binary_grad]  # List of one Tensor, since we have one input
