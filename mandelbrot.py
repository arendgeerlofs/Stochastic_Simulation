import numpy as np

def complex_matrix(xmin, xmax, ymin, ymax, pixel_density):
    """
    Creating a matrix with complex values
    """
    re = np.linspace(xmin, xmax, int((xmax - xmin) * pixel_density))
    im = np.linspace(ymin, ymax, int((ymax - ymin) * pixel_density))
    return re[np.newaxis, :] + im[:, np.newaxis] * 1j

def mandelbrot_matrix(matrix, iterations):
    """
    Creating a matrix with ones for inside Mandelbrot and zeros outside
    """
    for i in range(np.shape(matrix)[0]):
        for j in range(np.shape(matrix)[1]):
            value = mandelbrot(matrix[i][j], iterations)
            matrix[i][j] = value
    return matrix

def mandelbrot(c, iterations):
    """
    Determining wheter complex value is inside Mandelbrot set or not
    """
    z = 0
    for _ in range(iterations):
        z = z**2 + c
    return abs(z) <= 2

def circular_domain(N):
    '''
    Creating a circular domain
    '''
    # Starting grid
    grid = np.zeros((N, N))
    
    # Finding centre
    centre = (N-1) / 2 
    
    # Finding radius
    R = N / 2 / 5
    
    # Checking whether gridpoint inside circle, if so put value to 1
    for j in range(N):
        for i in range(N):
            dif_x = abs((i) - centre)
            dif_y = abs((j) - centre)
            c = np.sqrt(dif_x ** 2 + dif_y ** 2) 
            if c < R:
                grid[j][i] = 1
                
    return grid

def MC_integration_circle(samples):
    '''
    Calculate the area of a circle usig Monte Carlo integration
    '''
    # Defining the radius of circle
    radius = 0.4
    total_area = (2*radius)**2
    
    count = 0
    
    # Sampling
    for k in range(samples):
        x = np.random.uniform(-radius,radius)
        y = np.random.uniform(-radius,radius)
        
        if x**2 + y**2 <= radius**2:
            count += 1
    
    return total_area*count/samples