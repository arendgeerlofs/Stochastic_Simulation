"""
Main function from which functions are called
"""

from functions import *

ITERATIONS = 100
SAMPLES = 100
RUNS = 100
types_of_sampling = ["RandomS", "LatinS", "OrthogS"]
data = stats_per_iteration_value(ITERATIONS, SAMPLES, RUNS, types_of_sampling)
t_values = t_test(data, RUNS)
print(t_values)

data_samples = stats_per_sample_value(ITERATIONS, SAMPLES, RUNS, types_of_sampling)
plot(data, a_m_est=True, difference=True, name="Difference_plot")
plot(data, a_m_est=True, name="Absolute_plot")
plot_std(data_samples, name="samples")
plot_f_test(data_samples, name="f_test")
