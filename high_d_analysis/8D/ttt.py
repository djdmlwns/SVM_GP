# %%
from pyDOE2 import bbdesign

sample = bbdesign(4)

sample[sample == 0.] = 0.5
sample[sample == -1.] = 0.
sample[sample == 1.] = 1.

#print(sample)

# %%
import sobol
sample = sobol.sample(4, 24)
print(sample)