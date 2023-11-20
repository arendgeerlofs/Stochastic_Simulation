"""
Function file for mandelbrot set, sampling methods,
MC integration, statistic tests and plots
"""

import numpy as np
import matplotlib.pyplot as plt
from mandelbrot import complex_matrix, mandelbrot

def mc_integration(iterations, samples, type_of_sampling):
    """
    Monte Carlo integration of mandelbrot set
    """

    # Set sample size
    xmin = -2
    xmax = 1
    ymin = -1
    ymax = 1
    total_area = abs(xmax - xmin)*abs(ymax-ymin)

    # Compute mandelbrot set
    matrix = complex_matrix(xmin, xmax, ymin, ymax, 100)
    mb_matrix = mandelbrot(matrix, iterations)

    # To save data
    ratio = np.zeros((1, 3))

    # For type of sampling compute ratio of samples in mandelbrot set
    if 'RandomS' in type_of_sampling:
        ratio[0][0] = random_sampling(mb_matrix, samples)

    if 'LatinS' in type_of_sampling:
        ratio[0][1] = latin_hypercube(mb_matrix, samples)

    if 'OrthogS' in type_of_sampling:
        ratio[0][2] = orthogonal_sampling(mb_matrix, samples)

    # Return estimate size of mandelbrot set
    return total_area*ratio

def random_sampling(mb_matrix, samples):
    """
    Compute random sampling samples and return the ration in mandelbrot set
    """
    counter = 0

    for _ in range(samples):
        # Get random integer in range of x and y indexes
        i = np.random.randint(0, np.shape(mb_matrix)[0]-1)
        j = np.random.randint(0, np.shape(mb_matrix)[1]-1)

        # Check if point in mandelbrot set
        if np.real(mb_matrix[i,j]):
            counter += 1

    return counter/samples

def latin_hypercube(mb_matrix, samples):
    """
    Compute Latin Hypercube sampling samples and return ratio in mandelbrot set
    """
    counter = 0

    if samples > min(np.shape(mb_matrix)[0], np.shape(mb_matrix)[1]):
        samples = min(np.shape(mb_matrix)[0], np.shape(mb_matrix)[1])

    # Create lists of all row and column indexes in sample set
    row_indexes = [v for v in range(np.shape(mb_matrix)[0])]
    column_indexes = [v for v in range(np.shape(mb_matrix)[1])]

    for i in range(samples):
        # Randomly choose row and column index
        i = np.random.choice(row_indexes)
        j = np.random.choice(column_indexes)
        # Check if random sample point in mandelbrot set
        if np.real(mb_matrix[i,j]):
            counter += 1
        # Remove row and column index from the index lists
        row_indexes.remove(i)
        column_indexes.remove(j)

    return counter/samples

def orthogonal_sampling(mb_matrix, samples):
    """
    Compute orthongal sampling samples and return ratio in mandelbrot set
    """
    if samples > np.shape(mb_matrix)[0] * np.shape(mb_matrix)[1]:
        samples = np.shape(mb_matrix)[0] * np.shape(mb_matrix)[1]

    # Calculate amount of major rows/columns
    major = int(np.ceil(np.sqrt(samples)))
    # Create array of Major x Major with increasing indexes ex:[[1, 2][3, 4]]
    x_array = np.arange(0, major**2, 1).reshape(major, major)
    y_array = np.arange(0, major**2, 1).reshape(major, major)
    # Transpose y array so permutation can be done over different axis
    y_array = y_array.T
    for i in range(major):
        # Permute array to introduce randomness in sampling
        x_array[i] = np.random.permutation(x_array[i])
        y_array[i] = np.random.permutation(y_array[i])
    # Transpose back
    y_array = y_array.T
    counter = 0
    for i in range(samples):
        # Transform x and y indexes to corresponding points in sample space
        x_index = int(x_array[int((i-i%major)/major)][i%major])
        y_index = int(y_array[int((i-i%major)/major)][i%major])
        # Test if point in mandelbrot set
        if np.real(mb_matrix[int(x_index*(np.shape(mb_matrix)[0]/major**2))]
                            [int(y_index*(np.shape(mb_matrix)[1]/major**2))]):
            counter += 1
    return counter/samples

def statistics(iterations, samples, runs, type_of_sampling):
    """
    Compute estimate mandelbrot area for different types of sampling
    """
    areas = np.zeros([runs, len(type_of_sampling)])
    for i in range(runs):
        # Run Monte Carlo integration for different hyperparameters
        areas[i] = mc_integration(int(iterations), samples, type_of_sampling)
    # Calculate mean, std and confidence interval over runs
    mean_area = np.mean(areas, axis=0)
    std_area = np.std(areas, axis=0)
    confidence_interval = np.percentile(areas, [2.5, 97.5], axis=0)
    return [mean_area, std_area, confidence_interval]

def stats_per_iteration_value(iterations, samples, runs, type_of_sampling):
    """
    Calculate mean, std and confidence intervals for different amounts of iterations
    """
    mean =[]
    conf_interval =[]
    data = np.zeros((iterations, 4, 3))
    for i in range(iterations):
        # Run statistics function for different hyperparameters
        mean, std, conf_interval =statistics(int(i+1), samples, runs, type_of_sampling)
        # Save data
        data[i,0] = mean
        data[i,1] = std
        data[i,2] = conf_interval[0]
        data[i,3] = conf_interval[1]
    return data

def stats_per_sample_value(iterations, samples, runs, type_of_sampling):
    """
    Calculate mean, std and confidence interval values for different amounts of samples
    """
    mean =[]
    conf_interval =[]
    data = np.zeros((int(samples/10), 4, 3))
    for i in range(0, samples, 10):
        # Run statistics functions for different hyperparameters
        mean, std, conf_interval =statistics(iterations, int(i+1), runs, type_of_sampling)
        # Save data
        data[int(i/10),0] = mean
        data[int(i/10),1] = std
        data[int(i/10),2] = conf_interval[0]
        data[int(i/10),3] = conf_interval[1]
    return data

def plot(data, a_m_est = False, difference = False, name="Test"):
    '''
    Plot and save the mean and confidence intervals of the area of the Mandelbrot set. 

    Input:
    data        = Collected data about Random, LHS and Orthogonal sampling 
    a_m_est     = True if you want the function to print the best estimation of the Mandelbrot area
    difference  = True if difference between current iteration area and best approximation area
    name        = string to change the filename of the saved plot
    '''
    iters = np.shape(data)[0]
    x_data = np.linspace(1,iters, iters)
    # Print estimated area of mandelbrot set
    if a_m_est:
        print(f'Estimated A_M with random sampling = {data[:,0][-1][0]}')
        print(f'Estimated A_M with latin hypercube = {data[:,0][-1][1]}')
        print(f'Estimated A_M with orthogonal sampling = {data[:,0][-1][2]}')
    plt.figure()
    # Plot difference between current iteration area and best estimation area
    if difference:
        plt.plot(x_data, data[:,0,0] - data[-1,0,0]*np.ones(iters),
                 'r', label = 'Random sampling')
        plt.plot(x_data, data[:,0,1] - data[-1,0,1]*np.ones(iters),
                 'y', label = 'LHS sampling')
        plt.plot(x_data, data[:,0,2] - data[-1,0,2]*np.ones(iters),
                 'b', label = 'Orthogonal sampling')
        plt.fill_between(x_data, data[:,2,0] - data[-1,0,0]*np.ones(iters),
                         data[:,3,0] - data[-1,0,0]*np.ones(iters), color = 'r', alpha= 0.5)
        plt.fill_between(x_data, data[:,2,1] - data[-1,0,1]*np.ones(iters),
                         data[:,3,1] - data[-1,0,1]*np.ones(iters), color = 'y', alpha= 0.5)
        plt.fill_between(x_data, data[:,2,2] - data[-1,0,2]*np.ones(iters),
                         data[:,3,2] - data[-1,0,2]*np.ones(iters), color = 'b', alpha= 0.5)
        plt.ylabel('Area difference')
        plt.title(f'Difference in estimated size between current iteration and {iters} iterations')
    # Plot current mean and 95%-confidence intervals
    else:
        plt.plot(x_data, data[:,0,0], label = 'Random sampling', color='red')
        plt.plot(x_data, data[:,0,1], label = 'Latin hypercube', color='yellow')
        plt.plot(x_data, data[:,0,2], label = 'Orthogonal sampling', color='blue')
        plt.fill_between(x_data, data[:,2,0], data[:,3,0], color = 'red', alpha= 0.5)
        plt.fill_between(x_data, data[:,2,1], data[:,3,1], color = 'yellow', alpha= 0.5)
        plt.fill_between(x_data, data[:,2,2], data[:,3,2], color = 'blue', alpha= 0.5)
        plt.ylabel('Estimated area of Mandelbrot set')
        plt.title('Number of iterations against the area of the Mandelbrot set')
    plt.xlabel('Number of iterations')
    plt.legend()
    plt.savefig(f"figures/{name}.pdf", dpi=300)
    plt.show()

def plot_std(data, name="Test"):
    """
    Plot estimate mandelbrot area over different amounts of iterations
    """
    type_of_sampling = ["Random sampling", "Latin hypercube", "Orthogonal sampling"]
    colors = ['r', 'b', 'g']
    samples = np.shape(data)[0]
    # Plot mean and std deviation
    for i in range(np.shape(data)[2]):
        plt.plot(np.linspace(1, samples*10+1, samples), data[:, 0, i],
                 color=colors[i], label=type_of_sampling[i])
        plt.fill_between(np.linspace(1, samples*10+1, samples),
         data[:, 0, 0] - data[:, 1, 0], data[:, 0, i]+data[:, 1, i], color=colors[i], alpha=0.5)
    # Set plot parameters
    plt.legend()
    plt.title("Mandelbrot size estimation for different sampling methods")
    plt.xlabel("Number of samples")
    plt.ylabel("Estimated size of mandelbrot set")
    plt.savefig(f"figures/{name}.pdf", dpi=300)
    plt.show()

def plot_f_test(data, name="F_test"):
    """
    Plot F-statistics over different amounts of runs
    """
    samples = np.shape(data)[0]
    # Calculate F-values
    f_value_1 = data[:, 1, 0]**2 / data[:, 1, 1]**2
    f_value_2 = data[:, 1, 0]**2 / data[:, 1, 2]**2
    f_value_3 = data[:, 1, 1]**2 / data[:, 1, 2]**2
    # Plot F-values between sampling methods
    plt.plot(np.linspace(1, samples*10+1, samples), f_value_1, 'r.', label="Random vs LHS")
    plt.plot(np.linspace(1, samples*10+1, samples), f_value_2, 'b.', label="Random vs Orthogonal")
    plt.plot(np.linspace(1, samples*10+1, samples), f_value_3, 'g.', label="LHS vs Orthogonal")
    # Plot critical F-value
    plt.plot(np.linspace(1, samples*10+1, samples), np.full((samples), 1.15),
             'black', label="critical value")
    # Set plot labels, title, legend and save plot
    plt.title("F-tests between sampling methods")
    plt.xlabel("Samples")
    plt.ylabel("F-statistic")
    plt.legend()
    plt.savefig(f"figures/{name}.pdf", dpi=300)
    plt.show()

def t_test(data, runs):
    """
    Calculate Welch t-test between the 3 different sampling methods
    """
    # Only use the last data
    data = data[-1]
    # Calculate Welch t-value
    t1 = (data[0, 0] - data[0, 1]) / np.sqrt(data[1, 0]**2/runs + data[1, 1]**2/runs)
    t2 = (data[0, 0] - data[0, 2]) / np.sqrt(data[1, 0]**2/runs + data[1, 2]**2/runs)
    t3 = (data[0, 1] - data[0, 2]) / np.sqrt(data[1, 1]**2/runs + data[1, 2]**2/runs)
    return t1, t2, t3
