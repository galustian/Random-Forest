import numpy as np
cimport numpy as np
cimport cython
from numpy cimport ndarray, float64_t, int_t
from cython.parallel import prange

cdef double[:, ::1] bootstrap_data(double[:, ::1] data):
    return data[np.random.choice(data.shape[0], data.shape[0])]

cdef bool is_pure(double[:] Y):
    return np.unique(Y).size == 1

#@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double gini_index(double[:, ::1] left, double[:, ::1] right):
    cdef:
        double purity = 0.0
        double class_ratio
        double[:, ::1] split
        int[:] class_counts, unique_classes
        int total_classes, bi, i

    for bi in range(2):
        if bi == 0:
            split = left
        else:
            split = right
        
        class_counts = np.bincount(split[:, -1].astype('int'))
        unique_classes = np.array(np.nonzero(class_counts))[0]
        
        for i in range(unique_classes.shape[0]):
            class_ratio = <double>class_counts[unique_classes[i]] / split.shape[0]
            purity += class_ratio * (1 - class_ratio)
    
    return purity

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef dict get_best_split(double[:, ::1] data):
    cdef:
        double[:, ::1] left, right, b_left, b_right
        double split_point, b_split_point, gini, b_gini = 999.9
        int b_predictor, pred_i, row_i
    
    for pred_i in range(data.shape[1]-1): # -1 => last column is class
        # split data for every predictor value (for every row), to a greater than and smaller dataset
        for row_i in range(data.shape[0]):
            split_point = data[row_i][pred_i]
            left = data[data[:, pred_i] < split_point]
            right = data[data[:, pred_i] >= split_point]
            
            gini = gini_index(left, right)

            if gini < b_gini:
                b_gini, b_split_point, b_predictor = gini, split_point, pred_i
                b_left, b_right = left.copy(), right.copy()

    # TODO add proportional gini change for Variable-Importance measure
    return {'left': b_left, 'right': b_right, 'split_point': b_split_point, 'predictor': b_predictor, 'gini': b_gini}


cdef dict recurse_tree(dict node, int depth, int max_depth, int max_x):
    cdef:
        dict left_node, right_node
        int most_common_class
    
    # check branch depth
    if depth >= max_depth:
        most_common_class = np.bincount(node.left[:, -1]).argmax()
        node['left_node'] = {'end_node': True, 'y_hat': most_common_class}

        most_common_class = np.bincount(node.right[:, -1]).argmax()
        node['right_node'] = {'end_node': True, 'y_hat': most_common_class}
        return
    
    # check max_x & convergence
    if node.left.shape[0] <= max_x or is_pure(node.left[:, -1]):
        most_common_class = np.bincount(node.left[:, -1]).argmax()
        node['left_node'] = {'end_node': True, 'y_hat': most_common_class}
        return
    else:
        left_node = get_best_split(node.left)
        del node['left']
        node['left_node'] = left_node
        recurse_tree(node['left_node'], depth+1, max_depth, max_x)


    if node.right.shape[0] <= max_x or is_pure(node.right[:, -1]):
        most_common_class = np.bincount(node.right[:, -1]).argmax()
        node['right_node'] = {'end_node': True, 'y_hat': most_common_class}
        return
    else:
        right_node = get_best_split(node.right)
        del node['right']
        node['right_node'] = right_node
        recurse_tree(node['right_node'], depth+1, max_depth, max_x)

    # entire decision-tree
    return node


# labels have to be last column
cdef create_tree(double[:, ::1] data, int bootstrap_size, int max_depth, int max_x)
    cdef:
        list bagged_trees = []
        dict tree
        size_t i

    for i in range(bootstrap_size):        
        cdef dict root_node = get_best_split(bootstrap_data(data))
        tree = recurse_tree(root_node, 1, max_depth, max_x)
        bagged_trees.append(tree)
