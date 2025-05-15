import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ocena_istnosci_roznic import test, permutation_test



x = [0.8, 0.7, 0.8, 0.72]
y = [0.8, 0.81,0.8,0.8]


print(test(x, y))
print(permutation_test(x, y))


'''
a = []
b = []
print(test(a,b))
print(permutation_test(a,b))
'''