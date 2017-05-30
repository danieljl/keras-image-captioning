from keras import backend as K
from keras import optimizers


# A patch for bug #3859 / #5945: clipnorm doesn't work with Embedding
# Taken from https://github.com/fchollet/keras/pull/4915#issuecomment-303372138

def clip_norm(g, c, n):
    if c > 0:
        if K.backend() == 'tensorflow':
            import tensorflow as tf
            import copy
            condition = n >= c
            then_expression = tf.scalar_mul(c / n, g)
            else_expression = g

            if hasattr(then_expression, 'get_shape'):
                g_shape = copy.copy(then_expression.get_shape())
            elif hasattr(then_expression, 'dense_shape'):
                g_shape = copy.copy(then_expression.dense_shape)
            if condition.dtype != tf.bool:
                condition = tf.cast(condition, 'bool')
            g = K.tensorflow_backend.control_flow_ops.cond(
                condition, lambda: then_expression, lambda: else_expression)
            if hasattr(then_expression, 'get_shape'):
                g.set_shape(g_shape)
            elif hasattr(then_expression, 'dense_shape'):
                g._dense_shape = g_shape
        else:
            g = K.switch(n >= c, g * c / n, g)
    return g


optimizers.clip_norm = clip_norm
