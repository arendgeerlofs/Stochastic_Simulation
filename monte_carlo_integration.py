import numpy as np

def mc_int_estimate(iterations, samples):
    matrix = complex_matrix(-2, 1, -1, 1, 100)
    mandelbrot = mandelbrot_matrix(matrix, iterations)
    in_mandelbrot = 0
    for i in range(samples):
        if np.real(mandelbrot[np.random.randint(np.shape(mandelbrot)[0]), np.random.randint(np.shape(mandelbrot)[1])]):
            in_mandelbrot += 1
    return in_mandelbrot/samples * ((1- -2)* (1- - 1))