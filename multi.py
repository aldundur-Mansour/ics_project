import numpy as np

def read_matrix_from_file(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
        matrix = np.array([[int(num) for num in line.split()] for line in lines])
    return matrix

matrix1 = read_matrix_from_file("matrix1.txt")
matrix2 = read_matrix_from_file("matrix2.txt")

result = np.matmul(matrix1, matrix2)
print(result)