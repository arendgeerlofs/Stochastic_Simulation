import numpy as np
from mandelbrot import *
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({'font.size': 12})

def MC_integration(iterations, samples):
    """
    Using Monte Carlo simulation to approximate the area of the Mandelbrot set 
    and of the circle.
    """
    
    xmin = -2
    xmax = 2
    ymin = -2
    ymax = 2
    total_area = abs(xmax - xmin)*abs(ymax-ymin)
    
    matrix = complex_matrix(xmin, xmax, ymin, ymax, 100)
    mb_matrix = mandelbrot(matrix, iterations)
    
    circle_matrix = circular_domain(abs(xmax - xmin)*100)
    
    counter = 0
    counter_circle = 0
    
    # Sampling and counting
    for k in range(samples):
        i = np.random.randint(0, np.shape(mb_matrix)[0]-1)
        j = np.random.randint(0, np.shape(mb_matrix)[1]-1)
        
        if np.real(mb_matrix[i,j]) == 1:
            counter += 1
        
        if circle_matrix[i,j] == 1:
            counter_circle += 1
    
    # Approximating the areas
    area_MB = total_area*counter/samples
    area_circle = total_area*counter_circle/samples
            
    return area_MB, area_circle

def statistics_control_variates(iterations, samples, runs):
    """
    Performing the control variates method and finding the necessary statistics 
    of the control variates method
    """
    
    areas_MB = np.array([])
    areas_circle = np.array([])
    
    # Running multiple MC simulations
    for i in range(runs):
        area_MB, area_circle = MC_integration(iterations, samples)
        
        areas_circle = np.append(areas_circle, area_circle)
        areas_MB = np.append(areas_MB, area_MB)
    
    # Statistics of MB
    mean_MB = np.mean(areas_MB, axis=0)
    std_MB = np.std(areas_MB, axis=0)
    
    # Calculation of c
    covariance = np.cov(areas_MB, areas_circle)[0,1]
    variance_circle = np.std(areas_circle, axis=0)**2
    
    c = -covariance/variance_circle
    
    # New quantity
    mean_circle = (0.4**2)*np.pi
    new_quantity = areas_MB + c*(areas_circle - mean_circle)
    
    # Final statistics control variates
    mean_cv = np.mean(new_quantity, axis=0)
    std_cv = np.std(new_quantity, axis=0)
    
    # Normal test
    res_MB = stats.normaltest(areas_MB)
    res_cv = stats.normaltest(new_quantity)
    
    # F-test
    F_test = std_MB**2/std_cv**2
    
    # Levene's test
    lev = stats.levene(areas_MB, new_quantity)
    
    return [mean_MB, std_MB, mean_cv, std_cv, res_MB, res_cv, F_test, lev]
    
# Calculate results
mean_MB, std_MB, mean_cv, std_cv, res_MB, res_cv, F_test, lev = statistics_control_variates(100, 10**4, 10**4)
print("normal test MB:", res_MB)
print("normal test cv:", res_cv)
print(lev)
print(F_test)

# Plot results
fig, ax = plt.subplots(figsize=(4,8), dpi=300)

ax.errorbar([1,2], [mean_MB, mean_cv], yerr=[std_MB, std_cv], fmt='o')

ax.set(xlim=(0.5, 2.5), xticks=[1,2])
ax.set_xticklabels(['MC', 'CV MC'], rotation='vertical', fontsize=18)
ax.set_ylabel('Mean area')

plt.savefig("Control_variates.pdf")
plt.plot() 
