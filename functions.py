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

def MC_integration(iterations, samples, randomS=False, LatinS=False):
    
    xmin = -2
    xmax = 2
    ymin = -2
    ymax = 2
    total_area = abs(xmax - xmin)*abs(ymax-ymin)
    
    matrix = complex_matrix(xmin, xmax, ymin, ymax, 100)
    mb_matrix = mandelbrot(matrix, iterations)
    
    counter = 0
    
    if randomS:
        counter = random_sampling(mb_matrix, samples)
    
    if LatinS:
        counter, samples = Latin_hypercube(mb_matrix, samples)

    return total_area*counter/samples

def random_sampling(mb_matrix, samples):
    counter = 0
    
    for k in range(samples):
        i = np.random.randint(0, np.shape(mb_matrix)[0]-1)
        j = np.random.randint(0, np.shape(mb_matrix)[1]-1)
        
        if np.real(mb_matrix[i,j]) == 1:
            counter += 1
            
    return counter

def Latin_hypercube(mb_matrix, samples):
    counter = 0
    rows = []
    columns = []
    
    if samples > np.shape(mb_matrix)[0]*np.shape(mb_matrix)[1]:
        samples = np.shape(mb_matrix)[0]*np.shape(mb_matrix)[1]
        print("The amount of samples used, was corrected to", samples)
        
    while len(rows) < np.shape(mb_matrix)[0]:
        while len(columns) < np.shape(mb_matrix)[1]:
            i = np.random.randint(0, np.shape(mb_matrix)[0]-1)
            j = np.random.randint(0, np.shape(mb_matrix)[1]-1)
            
            if i not in rows and j not in columns:
                rows.append(i)
                columns.append(j)
                
                if np.real(mb_matrix[i,j]) == 1:
                    counter += 1
                    
    return counter, samples