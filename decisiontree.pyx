# distutils: language=c++

import numpy as np
cimport numpy as np
cimport cython
from numpy cimport ndarray, float64_t, int_t
from libcpp cimport bool
from libc.math cimport ceil, sqrt

cdef ndarray[float64_t, ndim=2] get_bootstrap(ndarray[float64_t, ndim=2] data):
    return data[np.random.choice(data.shape[0], data.shape[0]), :]

cdef bool is_pure(ndarray[int_t, ndim=1] Y):
    return np.unique(Y).size == 1

cdef add_endnode(dict node, bool left, bool right):
    cdef int most_common_class
    if left:
        most_common_class = np.bincount(node['left'][:, -1].astype('int')).argmax()
        node['left_node'] = {'end_node': True, 'y_hat': most_common_class}
        del node['left']
    if right:
        most_common_class = np.bincount(node['right'][:, -1].astype('int')).argmax()
        node['right_node'] = {'end_node': True, 'y_hat': most_common_class}
        del node['right']

#@cython.boundscheck(False)
#@cython.wraparound(False)
@cython.cdivision(True)
cpdef double gini_index(ndarray[float64_t, ndim=2] left, ndarray[float64_t, ndim=2] right):
    cdef:
        double purity = 0.0
        double class_ratio
        ndarray[float64_t, ndim=2] split
        ndarray[int_t, ndim=1] class_counts, unique_classes
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
cpdef dict get_best_split(ndarray[float64_t, ndim=2] data):
    cdef:
        ndarray[float64_t, ndim=2] left, right, b_left, b_right
        double split_point, b_split_point, gini, b_gini = 999.9
        int b_predictor, pred_i, predictor, row_i
        int num_random_predictors = <int>ceil(sqrt(data.shape[1]-1)) # size of random_predictors array
        ndarray[int_t] random_predictors = np.random.choice(data.shape[1]-1, num_random_predictors, replace=False)
    
    for pred_i in range(random_predictors.shape[0]):
        predictor = random_predictors[pred_i]
        # Split data 1. For every predictor value => For every row, to get best left / right split
        for row_i in range(data.shape[0]):
            split_point = data[row_i][predictor]
            left = data[data[:, predictor] < split_point]
            right = data[data[:, predictor] >= split_point]
            
            gini = gini_index(left, right)

            if gini < b_gini:
                b_gini, b_split_point, b_predictor = gini, split_point, predictor
                b_left, b_right = left.copy(), right.copy()

    # TODO add proportional gini change for Variable-Importance measure
    return {'left': b_left, 'right': b_right, 'split_point': b_split_point, 'predictor': b_predictor, 'gini': b_gini}


cdef dict recurse_tree(dict node, int depth, int max_depth, int max_x):
    cdef:
        dict left_node, right_node
    
    # CHECK Zero-Node left or right => both have same class
    if node['left'].shape[0] == 0 or node['right'].shape[0] == 0:
        if node['left'].shape[0] == 0:
            node['left'] = node['right']
        else:
            node['right'] = node['left']
        add_endnode(node, left=True, right=True)
        if depth == 1:
            return node
        return



    # CHECK branch depth
    if depth >= max_depth:
        add_endnode(node, left=True, right=True)
        return
    
    # CHECK max_x & convergence else: recurse further
    if node['left'].shape[0] <= max_x or is_pure(node['left'][:, -1].astype('int')):
        add_endnode(node, left=True, right=False)
    else:
        left_node = get_best_split(node['left'])
        del node['left']
        node['left_node'] = left_node
        recurse_tree(node['left_node'], depth+1, max_depth, max_x)


    if node['right'].shape[0] <= max_x or is_pure(node['right'][:, -1].astype('int')):
        add_endnode(node, right=True, left=False)
    else:
        right_node = get_best_split(node['right'])
        del node['right']
        node['right_node'] = right_node
        recurse_tree(node['right_node'], depth+1, max_depth, max_x)

    # entire decision-tree
    return node


# labels have to be last column
cpdef create_tree(ndarray[float64_t, ndim=2] data, int n_trees, int max_depth, int max_x):
    cdef:
        list bagged_trees = []
        dict tree, root_node
        size_t i

    for i in range(n_trees):        
        root_node = get_best_split(get_bootstrap(data))
        tree = recurse_tree(root_node, 1, max_depth, max_x)
        bagged_trees.append(tree)
    
    return bagged_trees
