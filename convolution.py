import numpy as np
import math


def hadamar_product(matrix1, matrix2):
    result = 0
    if np.shape(matrix1) != np.shape(matrix2): return "Shapes of matrices is different"
    else:
        result = sum([np.dot(matrix1[i][:], matrix2[i][:]) for i in iter(range(len(matrix1)))])
    return result


def stack_padding(matrix, padding: tuple, pad_value = 0) -> np.array:
    if isinstance(padding, tuple):
        height, width = np.shape(matrix)
        pad_col, pad_row = np.ones(height) * pad_value, np.ones(width + 2 * padding[0]) * pad_value
        if padding[0] == 1: col_block = np.reshape(pad_col, (height, 1)) 
        else: col_block = np.column_stack(tuple([pad_col for _ in range(padding[0])]))
        if padding[1] == 1: row_block = pad_row
        else: row_block = np.vstack(tuple([pad_row for _ in range(padding[1])]))
        output_matrix = np.hstack((col_block, matrix, col_block))
        output_matrix = np.vstack((row_block, output_matrix, row_block))
        return output_matrix 
    else: return "Padding must be a tuple in the following format: (int1, int2)!"


def add_padding(matrices, padding: tuple, pad_value = 0):
    if len(np.shape(matrices)) == 3:   #Case 1: 3-dimensional input image (matrices)
        output_matrices = []
        for matrix in matrices:
            new_matrix = stack_padding(matrix, padding, pad_value)
            output_matrices.append(new_matrix)
        
        return output_matrices
    else:   #Case 2: 2-dimensional input image (matrices)                  
        return stack_padding(matrices, padding, pad_value)


def conv_decorator(func):
    def wrapper(*args, **kwargs):
        padding, pad_val = kwargs.get('padding', (0, 0)), kwargs.get('pad_val', 0)
        if len(np.shape(args[0])) == 3:
            if len(np.shape(args[1])) == 3:
                #Case 1: 3-dimensional image and 3-dimensional kernel 
                conv_dim1 = np.shape(args[0])[0] - np.shape(args[1])[0] + 2 * padding[0] + 1
                conv_dim2 = np.shape(args[0])[1] - np.shape(args[1])[1] + 2 * padding[1] + 1 
                general_conv = np.zeros((conv_dim1, conv_dim2))
                for channel_num in iter(range(3)):
                    general_conv += func(args[0][: , : , channel_num], args[1][channel_num], padding, pad_val)        

                return general_conv
        
            elif len(np.shape(args[1])) == 2: 
                #Case 2: 3-dimensional image / tensor (height, width, channels) and 2-dimensional kernel.
                conv_maps = []
                for channel_num in iter(range(3)):
                    conv_maps.append(func(args[0][:, :, channel_num], args[1], padding, pad_val))
                
                return np.array(conv_maps) 
        
        else: return func(args[0], args[1], padding, pad_val) #Case 3: 2-dimensional image and kernel
    return wrapper



@conv_decorator
def get_convolution(matrix: np.array, filter_matrix: np.array, padding = (0, 0), pad_val = 0):
    '''
    Function applies the filter to the matrix and returns an activation map
    Size of activation map: matrix_size - filter_size + 1
    '''
    if padding != (0, 0):
        matrix = add_padding(matrix, padding, pad_val)
    
    matrix_shape, filter_shape = np.shape(matrix), np.shape(filter_matrix)
    act_map_size = matrix_shape[0] - filter_shape[0] + 1, matrix_shape[1] - filter_shape[1] + 1
    activation_map = [[] for _ in iter(range(act_map_size[0]))]
    for i in iter(range(act_map_size[0])):
        activation_map[i] = [hadamar_product(matrix[i : i + filter_shape[0], j : j + filter_shape[1]], 
                            filter_matrix) for j in iter(range(act_map_size[1]))]
    return np.array(activation_map)

