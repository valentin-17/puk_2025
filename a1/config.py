# Define parameters for the distributions
# Binomial Distributions
BUN_N = 1
BUN_P = 1
PATTY_N = 2
PATTY_P = 0.74
BACON_N = 3
BACON_P = 0.30
SALAD_N = 3
SALAD_P = 0.33
SAUCE_N = 3
SAUCE_P = 0.66

# Poisson Distributions
SPICE_MU = 0.98
CHEESE_MU = 0.90
VEGGIE_MU = 1.97

# Random state
SEED = 42

# Simulation Params
# Timings in seconds
SIM_TIME = 6 * 3600 # runs the simulation for 6 hours to process all orders
SIM_START = 11 * 3600 # 11 hours * 3600 seconds/hour
SIM_END = 14 * 3600

PICKUP_DELAY = 0.5 * 3600 # half an hour

MEAN_BETWEEN_ARRIVALS = 105
SCALE_BETWEEN_ARRIVALS = 8

# Other params
N_LINECOOKS = 2
N_ASSEMBLERS = 1