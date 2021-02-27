import numpy as np

# Global Model Assumptions
HORIZON = 20 # years, length of time the model covers.
year = np.arange(1,HORIZON+1) # an index for temporal calculations.
SAMPSIZE = 1000 # the number of iterations in the Monte Carlo simulation.
run = np.arange(1, SAMPSIZE+1) # the iteration index.
TAXRATE = 38 # %
DISCOUNTRATE = 12 # %/year, used for discounted cash flow calculations.
DEPRPER = 7 # years, the depreciation schedule for the capital.