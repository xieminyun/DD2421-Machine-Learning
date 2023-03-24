import numpy as np
import lab3 as lb


labels = np.array([[1], [2], [3], [3], [1]])
pr = lb.computePrior(labels)
print(pr)