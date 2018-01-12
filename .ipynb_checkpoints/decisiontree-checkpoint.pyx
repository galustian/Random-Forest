import numpy as np
cimport numpy as np
from numpy cimport ndarray, float64_t, int_t
cimport cython

'''int_dim1 = np.ndarray()
int_dim2 = np.ndarray()
float_dim1 = np.ndarray()
float_dim2 = np.ndarray()

ctypedef ndarray[int_t, ndim=1] int_dim1
ctypedef ndarray[int_t, ndim=2] int_dim2
ctypedef ndarray[float32_t, ndim=1] float_dim1
ctypedef ndarray[float32_t, ndim=2] float_dim2
'''

'''cdef double num_class_in_data(int class_, ndarray[np.float32_t, ndim=2] X):
    cdef int c, class_count = 0
    for c in X[:, -1]:
        if c == class_: class_count += 1
    return <double>class_count
'''

'''
cdef ndarray[int_t, ndim=2] unique_counts(int_dim1 classes):
    cdef ndarray[int_t, ndim=1] unique_counts = np.bincount(classes)
    cdef ndarray[int_t, ndim=1] unique_index = np.nonzero(unique_counts)[0]

    cdef ndarray[int_t, ndim=1] counts = np.array([], dtype=int_t)
    cdef int i_counts
    for i_counts in unique_counts:
        if i != 0:
            counts.append(i_counts)

    return np.array([unique_counts, counts], dtype=int_t)
'''
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double gini_index(ndarray[float64_t, ndim=2] left, ndarray[float64_t, ndim=2] right):
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
        
        class_counts = np.bincount(split[:, -1])
        unique_classes = np.nonzero(class_counts)
        total_classes = unique_classes.shape[0]
    
        for i in range(unique_classes.shape[0]):
            class_ratio = class_counts[unique_classes[i]] / total_classes
            purity += class_ratio * (1 - class_ratio)
    
    return purity

''''
cdef dict get_best_split(ndarray X, ndarray Y)
for class...
    for every data-point
        gini_index()

    # best_split
    return {'g_idx': g_idx, 'left': l_d, 'right': r_d}

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
