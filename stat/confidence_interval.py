import numpy as np
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m,h

### PRFNet ###

# data = [77.14,76.69,75.29,70.56] #n= 1

# data = [83.36,80.49,79.45,72.84] #n=5

# data = [83.67,81.48,79.97,72.89] #n=10

# data = [83.44,82.15,79.45,70.79] #n=50

# data = [82.19,81.32,78.04,68.62] #n=100

# data = [81.07,81.11,76.27,64.06] #n=200
### PRFNet ###



### PE-MIU ###

# data = [86.51,83.44,86.05,79.54,91.03,84.78] #n=1

# data = [89.93,86.82,88.60,83.08,92.39,87.91] #n=5

# data = [90.40,87.40,89.02,83.29,93.89,87.91] #n=10

# data = [90.75,86.90,88.86,83.08,94.43,89.67] #n=50

# data = [90.05,85.65,88.34,82.61,94.02,88.86] #n=100

data = [51.87,51.86,52.66] #n=200


###PE-MIU###

mean,confidence = mean_confidence_interval(data)

print(mean,confidence)

