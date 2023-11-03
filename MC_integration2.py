# Monte Carlo integration

import numpy as np
from mandelbrot import *
import matplotlib.pyplot as plt

def MC_integration(iterations, samples):
    
    xmin = -2
    xmax = 2
    ymin = -2
    ymax = 2
    total_area = abs(xmax - xmin)*abs(ymax-ymin)
    
    matrix = complex_matrix(xmin, xmax, ymin, ymax, 100)
    mb_matrix = mandelbrot(matrix, iterations)
    
    counter = 0
    
    for k in range(samples):
        i = np.random.randint(0, 399)
        j = np.random.randint(0, 399)
        
        if np.real(mb_matrix[i,j]) == 1:
            counter += 1
            
    return total_area*counter/samples

# # Checkig whether puting matrix into mandelbrot gives same result as
# # mandelbrot_matrix
# matrix = complex_matrix(-2, 2, -2, 2, 100)
# mb_matrix_A = mandelbrot(matrix, 100)
# mb_matrix_B = mandelbrot_matrix(matrix, 100)
# print(np.nonzero(mb_matrix_A-mb_matrix_B))


# Setting ranges for number of iterations and samples
iterations = [10*k for k in range(1,41)]
samples = [10**j for j in range(4, 7)]

# Plotting iterations on x-axis
for s in samples:
    area_list = np.array([])
    for i in iterations:
        area = MC_integration(i,s)
        area_list = np.append(area_list, area)
    plt.plot(iterations, area_list)

plt.legend([r'$N=10^4$', r'$N=10^5$', r'$N=10^6$'])
plt.xlabel('Number of iterations')
plt.ylabel('Area')
plt.show()

# Choosing 1 number of iterations and plotting sample sizes
iteration = 400
area_list = np.array([])

samples = [10*j for j in range(1, 10*10)]

for s in samples:
    area = MC_integration(iteration,s)
    area_list = np.append(area_list, area)
    
plt.semilogx(samples, area_list)
plt.xlabel('Sampels')
plt.ylabel('Area')
plt.show()
    
        