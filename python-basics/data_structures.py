import numpy as np
import pandas as pd

# Series data structure
# format => pd.Series(data, index=index)
# From ndarray
s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])
print(s)
# Accessing first pair
print("First value in series: ", s[0])
