from functions import *


iterations = 100
samples = 100
runs = 100

data = stats_per_iteration_value(iterations, samples, runs, type_of_sampling = ['RandomS', 'LatinS', 'OrthogS'])
t_values = t_test(data, runs)
print(t_values)

#data_samples = stats_per_sample_value(iterations, samples, runs, type_of_sampling = ['RandomS', 'LatinS', 'OrthogS'])
plot(data, A_M_est=True, difference=True, name="Difference_plot")
plot(data, A_M_est=True, name="Absolute_plot")
#plot_std(data_samples, name="samples")
#plot_f_test(data_samples, name="f_test")