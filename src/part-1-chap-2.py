# %%
# Chapter 2 - Setting Up the Analysis

# %%
# Deterministic Base Case
'''
ACME-Chem Company is considering the development of a chemical reactor and
production plant to deliver a new plastic compound to the market. The marketing department estimates that the market can absorb a total of 5 kilotons a year when it is mature, but it will take five years to reach that maturity from halfway through construction, which occurs in two phases.  They estimate that the market will bear a price of $6 per pound.

Capital spending on the project will be $20 million per year for each year of the first phase of development and $10 million per year for each year of the second phase.  Management estimates that development will last four years, two years for each phase. After that, a maintenance capital spending rate will be $2 million per year over the life of the operations. Assume a seven-year straight-line depreciation schedule for these capital expenditures. (To be honest, a seven-year schedule is too short in reality for these kinds of expenditures, but it will help illustrate the results of the depreciation routine better than a longer schedule.)

Production costs are estimated to be fixed at $3 million per year, but will escalate at 3% annually. Variable component costs will be $3.50 per pound, but cost reductions are estimated to be 5% annually. General, sales, and administrative (GS&A) overhead will be around 20% of sales.

The tax rate is 38%. The cost of capital is 12%. The analytic horizon is 20 years.

The problem is to determine the following:
1. Cash flow profile.
2. Cumulative cash flow profile.
3. Net present value (NPV) of the cash flow.
4. Pro forma table.
5. Sensitivity of the NPV to low and high changes in assumptions.
'''

# %%
# The Risk Layer
'''
The market intelligence group has just learned that RoadRunner Ltd. is developing a competitive product. Marketing believes that there is a 60% chance RoadRunner will also launch a production facility in the next four to six years. If they get to market before ACME-Chem, half the market share will be available to ACME-Chem. If they are later, ACME-Chem will maintain 75% of the market. In either case, the price pressure will reduce the monopoly market price by 15%.

What other assumptions should be treated as uncertainties in the base case?

The problem is to show the following:
1. Cash flow and cumulative cash flow with confidence bands.
2. Histogram of NPV.
3. Cumulative probability distribution of NPV.
4. Waterfall chart of the pro forma table line items.
5. Tornado sensitivity chart of 80th percentile ranges in uncertainties.
'''

# %%
# Imports
from PIL import Image
from IPython.display import display, Image
import pandas as pd
import numpy as np

# %%
# Influence Diagram
'''
Translate the context of the problem we have been asked to analyze to a type of flowchart that communicates the essence of the problem.  This flowchart is called an influence diagram.
'''
display(Image(filename='../assets/fig-2-1-influence-diagram-deterministic-base-case.png'))

# %%
# Import Global Model Assumptions
from global_assumptions import *
'''
HORIZON = 20 # years, length of time the model covers.
year = np.arange(1,HORIZON+1) # an index for temporal calculations.
SAMPSIZE = 1000 # the number of iterations in the Monte Carlo simulation.
run = np.arange(1, SAMPSIZE+1) # the iteration index.
TAXRATE = 38 # %
DISCOUNTRATE = 12 # %/year, used for discounted cash flow calculations.
DEPRPER = 7 # years, the depreciation schedule for the capital.
'''

print(HORIZON)
print(SAMPSIZE)
print(TAXRATE)
print(DISCOUNTRATE)
print(DEPRPER)

# %%
# Read in risk assumptions data
d_data = pd.read_csv('../data/risk_assumptions.csv')
d_data['variable'] = d_data['variable'].str.replace('.','_')

# %%
# Slice the p50 values from dataframe d_data, and assign p50 values to variables
d_vals = d_data[['variable','p50']][:13].set_index('variable').T
print(d_vals)
p1_dur = d_vals['p1_dur'].values
p2_dur = d_vals['p2_dur'].values
p1_capex = d_vals['p1_capex'].values
p2_capex = d_vals['p2_capex'].values
maint_capex = d_vals['maint_capex'].values

# %%
# CAPEX Module
phase = (year <= p1_dur) * 1 + \
        ((year > p1_dur) & (year <= (p1_dur + p2_dur))) * 2 + \
        (year > (p1_dur + p2_dur)) * 3

print(phase)

capex = (phase == 1) * p1_capex/p1_dur + \
        (phase == 2) * p2_capex/p2_dur + \
        (phase == 3) * maint_capex

print(capex)

# %%
# Generate depreciation matrix
depr_matrix = year.repeat(HORIZON).reshape(HORIZON,HORIZON).transpose()
print(depr_matrix)

# %%
# FINISHED PAGE 22