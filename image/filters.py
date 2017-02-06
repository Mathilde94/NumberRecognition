import math
import numpy as np

SIZE = 3

def gaussian(x, y, sigma=3):
    result = math.exp(-1 / float(2 * sigma**2) * float(x*x + y*y)) / float(2 * math.pi * sigma**2)
    return result


def gaussian_filter(np_array, sigma=1):
    output = np.zeros(np_array.shape)
    gaus_matrix = gaussian_matrix(sigma=sigma)
    k = SIZE

    for x in range(output.shape[0]):
        for y in range(output.shape[1]):
            g = 0
            for u in range(-k, k):
                for v in range(-k, k):
                    if x+u < 0 or x+u >= np_array.shape[0]:
                        continue
                    if y+v < 0 or y+v >= np_array.shape[1]:
                        continue
                    g += gaus_matrix[SIZE + u, SIZE + v] * np_array[x+u, y+v]
            output[x, y] = g
    return output


def gaussian_matrix(sigma=1):
    halfwidth = SIZE
    size = 2 * halfwidth + 1
    matrix = np.zeros((size, size))

    for i in range(-halfwidth, halfwidth):
        for j in range(-halfwidth, halfwidth):
            matrix[i+halfwidth, j+halfwidth] = gaussian(i, j, sigma=sigma)

    return matrix
