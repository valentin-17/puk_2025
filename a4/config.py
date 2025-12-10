# Simulation Params
SEED = 42
N_SIMS = 1

SIM_START = 0
SIM_END = 3 * 3600  # 3 hours in seconds
ARRIVAL_MEAN = 90

# routing: mapping burger_type -> list of machines to visit with their processing time variant
# format: burger_type: [(machine, variant), ...]
ROUTINGS = {
    'B1': [('M4', 'a'), ('M2', 'a'), ('M1', 'a'), ('M3', 'a'), ('M5', 'a')],
    'B2': [('M1', 'b'), ('M4', 'b'), ('M5', 'b'), ('M2', 'b'), ('M4', 'c'), ('M3', 'b')],
    'B3': [('M4', 'd'), ('M2', 'c'), ('M5', 'c'), ('M1', 'c')],
    'B4': [('M2', 'd'), ('M1', 'a'), ('M5', 'd'), ('M3', 'c'), ('M4', 'e')],
}

# different processing times for machines depending on burger_types
# Format: machine: variant: (dist, param1, param2)
PROCESSING_TIME = {
    'M1': {
        'a': ('exponential', 70),
        'b': ('exponential', 30),
        'c': ('normal', 35, 10),
    },
    'M2': {
        'a': ('normal', 10, 2),
        'b': ('uniform', 25, 45),
        'c': ('uniform', 10, 20),
        'd': ('uniform', 20, 30),
    },
    'M3': {
        'a': ('uniform', 6, 14),
        'b': ('exponential', 60),
        'c': ('normal', 25, 5),
    },
    'M4': {
        'a': ('normal', 40, 5),
        'b': ('normal', 35, 10),
        'c': ('uniform', 30, 40),
        'd': ('normal', 100, 15),
        'e': ('uniform', 50, 60),
    },
    'M5': {
        'a': ('uniform', 5, 15),
        'b': ('exponential', 40),
        'c': ('normal', 20, 5),
        'd': ('normal', 40, 4),
    },
}