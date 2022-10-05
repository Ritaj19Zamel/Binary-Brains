import tensorflow as tf

def maybe_num_nodes(index: tf.Tensor, num_nodes:int = None):
    """
    tf.argmax(row).numpy() -> int 
    num_nodes -> int
    need type conversion 
    return tf.argmax(row).numpy() + 1 if num_nodes is None else num_nodes 
    """
    return tf.reduce_max(index).numpy() + 1 if num_nodes is None else int(num_nodes)


def degree(index, num_nodes= None,
           dtype = None):
    r"""Computes the (unweighted) degree of a given one-dimensional index
    tensor.

    Args:
        index (LongTensor): Index tensor.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dtype (:obj:`torch.dtype`, optional): The desired data type of the
            returned tensor.

    :rtype: :class:`Tensor`

    Example:

        >>> row = tf.constant([0, 1, 0, 2, 0])
        >>> degree(row, dtype=tf.int32)
        tensor([3., 1., 1.])
    """
    N = maybe_num_nodes(index, num_nodes)
    out = tf.Variable(tf.zeros((N, ), dtype=tf.int32))
    one = tf.ones((tf.size(index).numpy(),), dtype=tf.int32)
    add = tf.compat.v1.scatter_add(out, index, one)
    return tf.cast(add,dtype=tf.float32)


