import numpy as np


# Given average losses
L = np.array([0.00010348635825769492, 1.568430761894922e-06, 0.2929140439763486, 5.819471089316412])  # Replace with actual values

# Compute factors
factors = 1 / L
factors /= factors.sum()

print(factors)
print(np.sum(factors))

for i in range(len(L)):
    print(L[i]*factors[i])

