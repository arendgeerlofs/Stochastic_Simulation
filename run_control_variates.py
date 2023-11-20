"""
Run control variates and print and plot data and statistics
"""

import matplotlib.pyplot as plt
from control_variates import stats_control_var

plt.rcParams.update({'font.size': 12})

# Calculate results
mean_mb, std_mb, mean_cv, std_cv, res_mb, res_cv, f_test, lev = stats_control_var(100, 10**4, 10**4)
print("normal test MB:", res_mb)
print("normal test cv:", res_cv)
print(lev)
print(f_test)

# Plot results
fig, ax = plt.subplots(figsize=(4,8), dpi=300)

ax.errorbar([1,2], [mean_mb, mean_cv], yerr=[std_mb, std_cv], fmt='o')

ax.set(xlim=(0.5, 2.5), xticks=[1,2])
ax.set_xticklabels(['MC', 'CV MC'], rotation='vertical', fontsize=18)
ax.set_ylabel('Mean area')

plt.savefig("Control_variates.pdf")
plt.plot()
