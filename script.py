import math

from tabulate import *
import random
from numba import cuda, float32
import numpy as np
import time

class Matrix:
    __ErrorCodes = {
        'rowLength' : ValueError("Rows must be of the same length"),
        'nonConformable' : ValueError("The matrices are non-conformable."),
        'sameDimensions' : ValueError("The matrices must be of the same size."),
        'operandTypes' : TypeError("Unsupported operand type(s)"),
    }

    @staticmethod
    def is_conformable(_matrix_a: 'Matrix', _matrix_b: 'Matrix') -> bool:
        return _matrix_a.Dimensions[1] == _matrix_b.Dimensions[0]

    def __init__(self, mat: list) -> None:
        # Instance Variables
        self._data: list[list[float]] = []
        self.Dimensions: tuple[int, int]

        row_length: int = len(mat[0])
        column_length: int = len(mat)
        for row in mat:
            if len(row) != row_length:
                raise self.__ErrorCodes['rowLength']

            self._data.append(row)

        self.Dimensions = (column_length, row_length) # the length of a column is how many rows there are and vice versa

    def _addition_abstract(self, other: 'Matrix') -> 'Matrix':
        if self.Dimensions != other.Dimensions:
            raise self.__ErrorCodes['sameDimensions']

        rows, cols = self.Dimensions
        resultant_list = [[0 for _ in range(cols)] for _ in range(rows)]

        for i in range(rows):
            for j in range(cols):
                resultant_list[i][j] = self._data[i][j] + other._data[i][j]

        return Matrix(resultant_list)

    def get_list(self):
        return self._data

    def __str__(self) -> str:
        return tabulate(
            self._data,
            tablefmt="fancy_grid"
        )

    def __add__(self, other: 'Matrix') -> 'Matrix':
        return self._addition_abstract(other)

    def __sub__(self, other: 'Matrix') -> 'Matrix':
        return self._addition_abstract(other)

    def __mul__(self, other) -> 'Matrix':
        if isinstance(other, (int, float)):
            return self.__rmul__(other)
        elif isinstance(other, Matrix):
            if not Matrix.is_conformable(self, other):
                raise self.__ErrorCodes['nonConformable']

            rows, cols = (self.Dimensions[0], other.Dimensions[1])
            resultant_list = [[0 for _ in range(cols)] for _ in range(rows)]

            run_length = self.Dimensions[1]

            a_run, b_run = 0, 0 # tracking positions of self and other (row, column)/(column, row)
            n = 0 # tracking current position of run

            while a_run < rows and b_run < cols:
                element_a = self._data[a_run][n]
                element_b = other._data[n][b_run]

                resultant_list[a_run][b_run] += element_a*element_b

                if n == run_length - 1:
                    n = 0
                    if b_run == cols - 1:
                        a_run += 1
                        b_run = 0
                        continue
                    b_run += 1
                    continue

                n += 1


            return Matrix(resultant_list)
        else:
            raise self.__ErrorCodes['operandTypes']

    def __rmul__(self, other) -> 'Matrix':
        if isinstance(other, (int, float)):
            resultant_list = [[e * other for e in row] for row in self._data]
            return Matrix(resultant_list)
        else:
            raise self.__ErrorCodes['operandTypes']

class NumbaMatrix(Matrix):
    def __mul__(self, other: 'NumbaMatrix') -> 'NumbaMatrix':
        if isinstance(other, (int, float)):
            return self.__rmul__(other)
        elif isinstance(other, NumbaMatrix):
            return self._cuda_matmul(other)
        else:
            raise self.__ErrorCodes['operandTypes']

    def __rmul__(self, other: (int, float)) -> 'NumbaMatrix':
        if isinstance(other, (int, float)):
            resultant_list = [[e * other for e in row] for row in self._data]
            return NumbaMatrix(resultant_list)
        else:
            raise self.__ErrorCodes['operandTypes']

    def _cuda_matmul(self, other: 'NumbaMatrix', use_shared=False):
        A = np.array(self._data, dtype=np.float32)
        B = np.array(other._data, dtype=np.float32)

        rows, cols = (self.Dimensions[0], other.Dimensions[1])
        C = np.zeros((rows, cols), dtype=np.float32)

        d_A = cuda.to_device(A)
        d_B = cuda.to_device(B)
        d_C = cuda.to_device(C)

        TILE_SIZE = 16
        threads_per_block = (TILE_SIZE, TILE_SIZE)
        blocks_per_grid_x = math.ceil(cols / TILE_SIZE)
        blocks_per_grid_y = math.ceil(rows / TILE_SIZE)
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        if use_shared:
            matmul_shared_kernel[blocks_per_grid, threads_per_block](d_A, d_B, d_C)
        else:
            matmul_kernel[blocks_per_grid, threads_per_block](d_A, d_B, d_C)

        result_array = d_C.copy_to_host()
        result_list = result_array.tolist()

        return NumbaMatrix(result_list)

@cuda.jit
def matmul_kernel(A, B, C):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if row < C.shape[0] and col < C.shape[1]:
        accumulator = 0.0
        for k in range(A.shape[1]):
            accumulator += A[row, k] * B[k, col]
        C[row, col] = accumulator

@cuda.jit
def matmul_shared_kernel(A, B, C):
    TILE_SIZE = 16

    sA = cuda.shared.array(shape=(16, 16), dtype=float32)
    sB = cuda.shared.array(shape=(16, 16), dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    row = by * cuda.blockDim.y + ty
    col = bx * cuda.blockDim.x + tx

    accumulator = 0.0

    for tile in range((A.shape[1] + TILE_SIZE - 1) // TILE_SIZE):
        if row < A.shape[0] and tile * TILE_SIZE + tx < A.shape[1]:
            sA[ty, tx] = A[row, tile * TILE_SIZE + tx]
        else:
            sA[ty, tx] = 0.0

        if tile * TILE_SIZE + ty < B.shape[0] and col < B.shape[1]:
            sB[ty, tx] = B[tile * TILE_SIZE + ty, col]
        else:
            sB[ty, tx] = 0.0

        cuda.syncthreads()

        for k in range(TILE_SIZE):
            accumulator += sA[ty, k] * sB[k, tx]

        cuda.syncthreads()

    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = accumulator

def generate_matrix(dimensions: (int, int)) -> Matrix:
    resultant_list = (np.random.randint(-20, 21, size=(dimensions[1], dimensions[0]))).tolist()
    return Matrix(resultant_list)

def generate_numba_matrix(dimensions: (int, int)) -> Matrix:
    resultant_list = (np.random.randint(-20, 21, size=(dimensions[1], dimensions[0]))).tolist()
    return NumbaMatrix(resultant_list)

"""
if __name__ == '__main__':
    data_list : list = [
        # (size , time)
    ]
    initial_size : int = 5
    no_iterations : int = 50
    increment : int = 5

    _current_iteration = 0
    _current_size = initial_size
    while _current_iteration < no_iterations:
        matrix_a = generate_matrix((_current_size, _current_size))
        matrix_b = generate_matrix((_current_size, _current_size))

        start_time = time.time()
        result = matrix_a * matrix_b
        end_time = time.time()

        data_list.append((_current_size, round(end_time - start_time, 4)))
        _current_size += increment
        _current_iteration += 1

    print(tabulate(data_list, headers=["Dimensions","Time"], tablefmt="pipe"))
"""

def test_matrix_time(initial_size : int, no_iterations : int, size_increment : int) -> list[tuple[int, float]]:
    data_list : list[tuple[int, float]] = []

    _current_iteration = 0
    _current_size = initial_size
    while _current_iteration < no_iterations:
        matrix_a = generate_matrix((_current_size, _current_size))
        matrix_b = generate_matrix((_current_size, _current_size))

        start_time = time.time()
        result = matrix_a * matrix_b
        end_time = time.time()

        data_list.append((_current_size, round(end_time - start_time, 4)))
        _current_size += size_increment
        _current_iteration += 1

    return data_list

if __name__ == '__main__':
    MatrixOne = generate_numba_matrix((25000, 25000))
    MatrixTwo = generate_numba_matrix((25000, 25000))
    Result = MatrixOne * MatrixTwo
    np.savetxt('result.txt', Result.get_list(), fmt='%.2f')
    print('done')