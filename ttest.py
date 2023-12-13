import numpy as np
import scipy.stats as sc

# t-test
greg_class  = np.array([87, 86, 86, 85, 79, 78, 78, 78, 77, 77])
peter_class = np.array([85, 72, 69, 67, 65, 60, 40, 35, 34, 33])

ttest = sc.ttest_ind(greg_class, peter_class, equal_var=False)
#print(ttest)

# Rescaled
peter_rs = (peter_class / 5.2) + 70.654

peter_z = (((72/5.2)+70.654) - np.mean(peter_rs)) / np.std(peter_rs)
greg_z  = (85 - np.mean(greg_class)) / np.std(greg_class)

ttest = sc.ttest_ind(greg_class, peter_rs, equal_var=True)

print(ttest)
print(f"Peter's z-score: {peter_z}\nGreg's z-score: {greg_z}")

