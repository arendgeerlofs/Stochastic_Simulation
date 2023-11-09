import numpy as np
import matplotlib.pyplot as plt

def complex_matrix(xmin, xmax, ymin, ymax, pixel_density):
    re = np.linspace(xmin, xmax, int((xmax - xmin) * pixel_density))
    im = np.linspace(ymin, ymax, int((ymax - ymin) * pixel_density))
    return re[np.newaxis, :] + im[:, np.newaxis] * 1j

def mandelbrot_matrix(matrix, iterations):
    for i in range(np.shape(matrix)[0]):
        for j in range(np.shape(matrix)[1]):
            value = mandelbrot(matrix[i][j], iterations)
            matrix[i][j] = value
    return matrix

def mandelbrot(c, iterations):
    z = 0
    for _ in range(iterations):
        z = z**2 + c
    return abs(z) <= 2

def MC_integration(iterations, samples, randomS=False, LatinS=False, Orthog=False):
    
    xmin = -2
    xmax = 2
    ymin = -2
    ymax = 2
    total_area = abs(xmax - xmin)*abs(ymax-ymin)
    
    # TODO choose pixel density/add to function variables
    matrix = complex_matrix(xmin, xmax, ymin, ymax, 100)
    mb_matrix = mandelbrot(matrix, iterations)
    
    counter = 0
    
    if randomS:
        counter = random_sampling(mb_matrix, samples)
    
    if LatinS:
        counter, samples = Latin_hypercube(mb_matrix, samples)

    if Orthog:
        counter, samples = Orthogonal_sampling(mb_matrix, samples)
    
    return total_area*counter/samples

def random_sampling(mb_matrix, samples):
    counter = 0
    
    for k in range(samples):
        i = np.random.randint(0, np.shape(mb_matrix)[0]-1)
        j = np.random.randint(0, np.shape(mb_matrix)[1]-1)
        
        if np.real(mb_matrix[i,j]):
            counter += 1
    print(counter)
    return counter

def Latin_hypercube(mb_matrix, samples):
    counter = 0
    
    if samples > max(np.shape(mb_matrix)[0], np.shape(mb_matrix)[1]):
        samples = max(np.shape(mb_matrix)[0], np.shape(mb_matrix)[1])
        print("The amount of samples used, was corrected to", samples)
    

    row_indexes = [v for v in range(np.shape(mb_matrix)[0])]
    column_indexes = [v for v in range(np.shape(mb_matrix)[1])]
    for i in range(samples):
        i = np.random.choice(row_indexes)
        j = np.random.choice(column_indexes)
        if np.real(mb_matrix[i,j]):
                counter += 1
        row_indexes.remove(i)
        column_indexes.remove(j)
    print(counter)
    return counter, samples

def Orthogonal_sampling(mb_matrix, samples):
    if samples > np.shape(mb_matrix)[0] * np.shape(mb_matrix)[1]:
        samples = np.shape(mb_matrix)[0] * np.shape(mb_matrix)[1]
        print("The amount of samples used, was corrected to", samples)
    major = int(np.ceil(np.sqrt(samples)))
    x_array = np.arange(0, major**2, 1).reshape(major, major)
    y_array = np.arange(0, major**2, 1).reshape(major, major)
    # Transpose so permutation over different axis
    y_array = y_array.T
    for i in range(major):
        x_array[i] = np.random.permutation(x_array[i])
        y_array[i] = np.random.permutation(y_array[i])
    # Transpose back
    y_array = y_array.T
    counter = 0
    for i in range(samples):
        x_index = int(x_array[int((i-i%major)/major)][i%major])
        y_index = int(y_array[int((i-i%major)/major)][i%major])
        if np.real(mb_matrix[int(x_index*(np.shape(mb_matrix)[0]/major**2))][int(y_index*(np.shape(mb_matrix)[1]/major**2))]):
            counter += 1
    return counter, samples
