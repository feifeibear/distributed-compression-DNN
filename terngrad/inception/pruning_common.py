from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def pruning_gradients(grads_and_vars, percent, residual_grads):
    """
    pruning grads according to the percent.
    """
    gradients, variables = zip(*grads_and_vars)
    pruned_gradients = []
    new_residual_grads = []
    current_start_pos = 0
    for gradient in gradients:
        if gradient is None:
            pruned_gradients.append(None)
            continue

        # find the top percent largest value.
        gradient_shape = tf.shape(gradient)
        gradient_flat = tf.reshape(gradient, [-1])
        grad_size = gradient_flat.shape.as_list()[0]
        print('FJR DEBUG in pruning_common grad_size ', grad_size)
        residual_grad = residual_grads[current_start_pos : current_start_pos + grad_size]
        current_start_pos = current_start_pos + grad_size
        gradient_flat = tf.add(gradient_flat, residual_grad)
        #size = tf.size(gradient_flat)
        k = int(grad_size * percent)
        #print(k)
        values,_ = tf.nn.top_k( gradient_flat, k=k, sorted=True, name=None )

        # set the values less than threshold in tensor to 0.
        threshold = values[-1]
        #print(threshold)
        zeros = tf.zeros(shape=tf.shape(gradient), dtype=tf.float32)
        where_cond = tf.reshape( tf.less(threshold, gradient), gradient_shape )
        pruned_gradient = tf.where(where_cond, gradient, zeros)
        pruned_gradients.append(pruned_gradient)
        new_residual_grads.append(tf.reshape(tf.subtract(gradient, pruned_gradient), [-1]))
    return list(zip(pruned_gradients, variables)), new_residual_grads
