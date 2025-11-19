"""
Planning and Configuration: Practical Task 1

Scenario:
  Students can order a burger online. The warm ingredients for this burger are then prepared by the 'Vorarbeitstheke'.
  The 'Burgermeister' assembles the warm and cold ingredients at the 'Zubereitungstheke' and puts the finished burger
  into the 'Auslage' where the burger waits to be picked up. Ingredient containers are considered resources that have
  to be refilled from an infinite stock. Workers are considered resources. Burger orders are artificially created based
  on previously derived distributions (see data_analysis.ipynb).
"""
import random
from typing import NamedTuple

import numpy as np
import simpy

from a1.config import (
    SEED,
    N_LINECOOKS,
    N_ASSEMBLERS,
    SIM_END,
    SIM_START,
    MEAN_BETWEEN_ARRIVALS,
    SCALE_BETWEEN_ARRIVALS, SIM_TIME, PICKUP_DELAY
)


class Burgenerator(NamedTuple):
    """Holds the metadata for Theke 3 during a simulation run."""
    prep: simpy.Resource
    assembly: simpy.Resource
    order_stats: list = []
    prep_usage: list = []
    assembly_usage: list = []

def _incoming_orders(env, burgenerator):
    """Generates orders from SIM_START until SIM_END."""
    order_window = SIM_END - SIM_START
    while env.now < order_window:
        burger = _sample_burger()
        env.process(_burger_process(env, burgenerator, burger))

        interarrival_time = np.random.normal(MEAN_BETWEEN_ARRIVALS, SCALE_BETWEEN_ARRIVALS)
        yield env.timeout(max(0, interarrival_time))

def _sample_burger():
    """Samples random burger based off of the distributions determined in data_analysis.ipynb."""
    pass

def _burger_process(env, burgenerator):
    order_time = env.now
    pickup_time = order_time + PICKUP_DELAY
    rework_count = 0

def _analyze_results(env):
    print('Simulation Results:')

def main():
    random.seed(SEED)
    env = simpy.Environment()
    burgenerator = Burgenerator(
        prep=simpy.Resource(env, capacity=N_LINECOOKS),
        assembly=simpy.Resource(env, capacity=N_ASSEMBLERS)
    )
    env.process(incoming_orders(env, burgenerator))
    env.run(until=SIM_TIME)

    analyze_results(env)

if __name__ == '__main__':
    main()