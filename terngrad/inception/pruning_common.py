from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def pruning_gradients(grads_and_vars, percent, lr):
    """
    pruning grads according to the percent.
    """
    gradients, variables = zip(*grads_and_vars)
    pruned_gradients = []
    residual_variables = []
    for gradient, variable in zip(gradients, variables):
        if gradient is None:
            pruned_gradients.append(None)
            continue

        # find the top percent largest value.
        # gradient_shape = tf.shape(gradient)
        gradient_flat = tf.reshape(gradient, [-1])
        size = gradient_flat.get_shape().as_list()
        k = int(size[0]*percent)
        #print(k)
        values,_ = tf.nn.top_k( gradient_flat, k=k, sorted=True, name=None )

        # set the values less than threshold in tensor to 0.
        threshold = values[-1]
        #print(threshold)
        zeros = tf.zeros(shape=tf.shape(gradient), dtype=tf.float32)
        where_cond = tf.less(threshold, gradient)
        pruned_gradient = tf.where(where_cond, gradient, zeros) 
        pruned_gradients.append(pruned_gradient)
#        residual_variables.append(tf.substract(variable, tf.multiply(-lr, tf.substract(gradient, pruned_gradients)))
        residual_variables.append(variable)
    return list(zip(pruned_gradients, residual_variables))
