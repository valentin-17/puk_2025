# Parameters for the distributions
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

# Simulation Parameters
# Random state
SEED = 1

# Timings in seconds
SIM_START = 11 * 3600 # 11 hours * 3600 seconds/hour
SIM_END = 14 * 3600
ORDER_TIME = 3 * 3600 # time to place orders is 3 hours

PICKUP_DELAY = 0.5 * 3600 # half an hour

TIME_BETWEEN_ARRIVALS_MEAN = 105
TIME_BETWEEN_ARRIVALS_SCALE = 8
WARM_PROCESSING_TIME_LOW = 360
WARM_PROCESSING_TIME_HIGH = 600
FREEZER_PREP_TIME_SHAPE = 10
FREEZER_PREP_TIME_SCALE = 2
TOASTING_TIME = 40 # seconds
ASSEMBLY_TIME_PER_INGREDIENT_MEAN = 5
ASSEMBLY_TIME_PER_INGREDIENT_SCALE = 1
PACKING_TIME_LOW = 10
PACKING_TIME_HIGH = 20
PACKING_TIME_FRIES_LOW = 15
PACKING_TIME_FRIES_HIGH = 30
REFILL_TIME_MEAN = 3 * 60 # given minutes scale, needs transform into seconds scale
REFILL_TIME_SCALE = 0.5 * 60 # given minutes scale, needs transform into seconds scale

# Other params
N_LINECOOKS = 2
N_ASSEMBLERS = 1
N_HELPERS = 1

MIN_INGREDIENTS = 2
MAX_INGREDIENTS = 20

FAIL_PROB = 0.05
FRIES_PROB = 0.5

N_SIMS = 500