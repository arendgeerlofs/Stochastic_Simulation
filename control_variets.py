import numpy as np
from mandelbrot import *

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

def MC_integration_circle(samples):
    
    radius = 2
    total_area = (2*radius)**2
    
    count = 0
    
    for k in range(samples):
        x = np.random.uniform(-radius,radius)
        y = np.random.uniform(-radius,radius)
        
        if x**2 + y**2 <= radius**2:
            count += 1
    
    return total_area*count/samples

def statistics_control_variets(iterations, samples, runs):
    
    areas_MB = np.array([])
    areas_circle = np.array([])
    
    for i in range(runs):
        area_circle = MC_integration_circle(samples)
        areas_circle = np.append(areas_circle, area_circle)
        
        area_MB = MC_integration(iterations, samples)
        areas_MB = np.append(areas_MB, area_MB)
    
    # Calculation of c
    covariance = np.cov(areas_MB, areas_circle)[0,1]
    variance_circle = np.std(areas_circle, axis=0)**2
    
    c = -covariance/variance_circle
    
    # New quantity
    mean_circle = 4*np.pi
    new_quantity = areas_MB + c*(areas_circle - mean_circle)
    
    # Final statistics
    mean = np.mean(new_quantity, axis=0)
    std = np.std(new_quantity, axis=0)
    confidence_interval = np.percentile(new_quantity, [2.5, 97.5], axis=0)
    
    return [mean, std, confidence_interval]

def statistics_MB(iterations, samples, runs):
    
    areas_MB = np.array([])
    
    for i in range(runs):
        area_MB = MC_integration(iterations, samples)
        areas_MB = np.append(areas_MB, area_MB)
    
    mean = np.mean(areas_MB, axis=0)
    std = np.std(areas_MB, axis=0)
    confidence_interval = np.percentile(areas_MB, [2.5, 97.5], axis=0)
    
    return [mean, std, confidence_interval]
    
    
a = statistics_control_variets(100, 10**4, 100)
b = statistics_MB(100, 10**4, 100)
print(a[0:2])
print(b[0:2])
