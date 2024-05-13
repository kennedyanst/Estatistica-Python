import numpy as np
from scipy import stats
import seaborn as sns

# Permutação

import math

math.factorial(3)

math.factorial(36) / math.factorial(36 - 5)

math.pow(36, 5)

# Combinação

math.factorial(6) / (math.factorial(2) * math.factorial(6-2))

math.factorial(6+2-1) / (math.factorial(2) * math.factorial(6-1))

# Interseção, união e diferença

# Interseção
a = (0,1,2,3,4,5,6,7)
b = (0,2,4,6,8)

set(a) and set(b)

# União
set(a) or set(b)

# Diferença
set(a).difference(set(b))
set(b).difference(set(a))