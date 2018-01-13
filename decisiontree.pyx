import numpy as np
cimport numpy as np
cimport cython
from numpy cimport ndarray, float64_t, int_t


#@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double gini_index(ndarray[float64_t, ndim=2] left, ndarray[float64_t, ndim=2] right):
    cdef:
        double purity = 0.0
        ndarray[float64_t, ndim=2] split
        ndarray[int_t, ndim=1] class_counts
        ndarray[int_t, ndim=1] unique_classes
        double class_ratio
        int total_classes, i, bi

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

#@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef dict get_best_split(ndarray[float64_t, ndim=2] data):
    cdef:
        ndarray[float64_t, ndim=2] left, right, b_left, b_right
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

    return {'left': b_left, 'right': b_right, 'split_point': b_split_point, 'predictor': b_predictor, 'gini': b_gini}


''''
cdef recurse_tree(dict node)

cdef create_tree(ndarray X, ndarray Y)
    cdef dict split = get_best_split(X, Y)
    cdef dict root_node = {
        'gini': split['g_idx'],
        'left_node': {'data': split['left']}
        'right_node': {'data': split['left']}
    }

recurse_tree(root_node['left_node'])
recurse_tree(root_node['right_node'])
'''
