"""
Compute statistics about control variates
"""

import numpy as np
from scipy import stats
from mandelbrot import complex_matrix, mandelbrot, circular_domain

def mc_integration(iterations, samples):
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
    for _ in range(samples):
        i = np.random.randint(0, np.shape(mb_matrix)[0]-1)
        j = np.random.randint(0, np.shape(mb_matrix)[1]-1)

        if np.real(mb_matrix[i,j]) == 1:
            counter += 1

        if circle_matrix[i,j] == 1:
            counter_circle += 1

    # Approximating the areas
    area_mb = total_area*counter/samples
    area_circle = total_area*counter_circle/samples

    return area_mb, area_circle

def stats_control_var(iterations, samples, runs):
    """
    Performing the control variates method and finding the necessary statistics 
    of the control variates method
    """

    areas_mb = np.array([])
    areas_circle = np.array([])

    # Running multiple MC simulations
    for _ in range(runs):
        area_mb, area_circle = mc_integration(iterations, samples)

        areas_circle = np.append(areas_circle, area_circle)
        areas_mb = np.append(areas_mb, area_mb)

    # Statistics of MB
    mean_mb = np.mean(areas_mb, axis=0)
    std_mb = np.std(areas_mb, axis=0)

    # Calculation of c
    covariance = np.cov(areas_mb, areas_circle)[0,1]
    variance_circle = np.std(areas_circle, axis=0)**2

    c = -covariance/variance_circle

    # New quantity
    mean_circle = (0.4**2)*np.pi
    new_quantity = areas_mb + c*(areas_circle - mean_circle)

    # Final statistics control variates
    mean_cv = np.mean(new_quantity, axis=0)
    std_cv = np.std(new_quantity, axis=0)

    # Normal test
    res_mb = stats.normaltest(areas_mb)
    res_cv = stats.normaltest(new_quantity)

    # F-test
    f_test = std_mb**2/std_cv**2

    # Levene's test
    lev = stats.levene(areas_mb, new_quantity)

    return [mean_mb, std_mb, mean_cv, std_cv, res_mb, res_cv, f_test, lev]
