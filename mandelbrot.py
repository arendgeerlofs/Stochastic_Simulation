"""
Mandelbrot functions
"""

import numpy as np

def complex_matrix(xmin, xmax, ymin, ymax, pixel_density):
    """
    Compute complex matrix based on sample space and density
    """
    # Real matrix
    re = np.linspace(xmin, xmax, int((xmax - xmin) * pixel_density))
    # Imaginary matrix
    im = np.linspace(ymin, ymax, int((ymax - ymin) * pixel_density))
    # Create complex matrix
    return re[np.newaxis, :] + im[:, np.newaxis] * 1j

def mandelbrot_matrix(matrix, iterations):
    """
    For each point in the matrix compute if it is in mandelbrot set
    """
    for i in range(np.shape(matrix)[0]):
        for j in range(np.shape(matrix)[1]):
            value = mandelbrot(matrix[i][j], iterations)
            matrix[i][j] = value
    return matrix

def mandelbrot(c, iterations):
    """
    Mandelbrot iteration function
    Check if certain point is in the mandelbrot set after certain amount of iterations
    """
    z = 0
    for _ in range(iterations):
        # Mandelbrot iteration function
        z = z**2 + c
    return abs(z) <= 2

def circular_domain(n):
    '''
    Creating a circular domain
    '''
    # Starting grid
    grid = np.zeros((n, n))

    # Finding centre
    centre = (n-1) / 2

    # Finding radius
    r = n / 2 / 5

    # Checking whether gridpoint inside circle, if so put value to 1
    for j in range(n):
        for i in range(n):
            dif_x = abs((i) - centre)
            dif_y = abs((j) - centre)
            c = np.sqrt(dif_x ** 2 + dif_y ** 2)
            if c < r:
                grid[j][i] = 1

    return grid
