"""
Planning and Configuration: Practical Task 2

Scenario:
  In addition to the scenario from task 1 here the storage has to be modeled as well. For this a new resource 'helper'
  is introduced. The helper can fetch new ingredients from the infinite storage to refill the finite storage bins of the
  prep station as well as the assembly station. The helper can only carry one ingredient per trip.

Implementation Details:
  This implementation is an extension of the implementation from task 1.

Insights:
"""
from collections import defaultdict

import numpy as np
import pandas as pd
import simpy

import a1.config as c


GEN = np.random.default_rng(c.SEED)


class Burgenerator:
    """Holds the metadata for Theke 3 during a simulation run."""
    def __init__(self, env: simpy.Environment):
        self.env = env
        self.prep: simpy.Resource = simpy.Resource(env, capacity=c.N_LINECOOKS)
        self.assembly: simpy.Resource = simpy.Resource(env, capacity=c.N_ASSEMBLERS)
        self.helper: simpy.Resource = simpy.Resource(env, capacity=c.N_HELPERS)
        self.usage_stats: defaultdict[str, list] = defaultdict(list)
        self.order_stats: defaultdict[str, list] = defaultdict(list)
        self.timeline_events = []

    def prepare(self):
        # we can assume that the prep_time is always freezer prep time + warm prep time since every burger has to have
        # at least one patty. the prep time is then independent of the total amount of warm ingredients.
        prep_time = (GEN.gamma(c.FREEZER_PREP_TIME_SHAPE, c.FREEZER_PREP_TIME_SCALE)
                     + GEN.uniform(c.WARM_PROCESSING_TIME_LOW, c.WARM_PROCESSING_TIME_HIGH))
        yield self.env.timeout(prep_time)
        #print(f'Warm ingredients prepared in {prep_time:.2f} seconds.')
        return prep_time

    def refill(self):
        refill_time = GEN.normal(c.REFILL_TIME_MEAN, c.REFILL_TIME_SCALE)
        yield self.env.timeout(refill_time)
        return refill_time

    def assemble(self, burger):
        # only count assembly time for the cold ingredients plus the toasting time, buns are also not considered a cold
        # ingredient as they have the 40-second toasting time
        assembly_time = (c.TOASTING_TIME
                         + (GEN.normal(c.ASSEMBLY_TIME_PER_INGREDIENT_MEAN, c.ASSEMBLY_TIME_PER_INGREDIENT_SCALE)
                         * sum(v for k, v in burger.items() if k not in ['bun', 'patty', 'bacon'])))
        yield self.env.timeout(assembly_time)
        #print(f'Burger assembled in {assembly_time:.2f} seconds.')
        return assembly_time

    def package(self):
        packing_time = GEN.uniform(c.PACKING_TIME_LOW, c.PACKING_TIME_HIGH)
        if GEN.random() < c.FRIES_PROB:
            packing_time += GEN.uniform(c.PACKING_TIME_FRIES_LOW, c.PACKING_TIME_FRIES_HIGH)
        yield self.env.timeout(packing_time)
        return packing_time

def _incoming_orders(env, burgenerator):
    """Generates orders from SIM_START until SIM_END."""
    order_window = c.SIM_END - c.SIM_START
    order_id = 0
    while env.now < order_window:
        burger = _sample_burger()
        env.process(_burger_process(env, burgenerator, burger, order_id))
        order_downtime = int(GEN.normal(c.TIME_BETWEEN_ARRIVALS_MEAN, c.TIME_BETWEEN_ARRIVALS_SCALE))
        order_id += 1
        yield env.timeout(max(0, order_downtime))

def _sample_burger():
    """Samples a random burger with MIN_INGREDIENTS and MAX_INGREDIENTS and at least one bun and one patty. Invalid burgers
    are immediately rejected."""
    while True:
        burger = {
            'bun': np.random.binomial(c.BUN_N, c.BUN_P),
            'patty': np.random.binomial(c.PATTY_N, c.PATTY_P),
            'bacon': np.random.binomial(c.BACON_N, c.BACON_P),
            'salad': np.random.binomial(c.SALAD_N, c.SALAD_P),
            'sauce': np.random.binomial(c.SAUCE_N, c.SAUCE_P),
            'spice': np.random.poisson(c.SPICE_MU),
            'cheese': np.random.poisson(c.CHEESE_MU),
            'vegetables': np.random.poisson(c.VEGGIE_MU),
        }
        total_ingredients = sum(burger.values())

        if (c.MIN_INGREDIENTS <= total_ingredients <= c.MAX_INGREDIENTS and
                burger['bun'] > 0 and
                burger['patty'] > 0):
            return burger

def _burger_process(env, burgenerator, burger, order_id):
    """Simulates a order process from incoming order to finished burger. This method uses the burgenerator class as the
    environment (i.e. Theke 3) in which the tasks are done. The individual tasks are modeled in the burgenerator class
    while inter-task activities like the failure of assembly or the wait to start the assembly are modeled here in this
    method. Further various timings and statistics are tracked to allow for later analysis."""
    order_time = env.now
    pickup_time = order_time + c.PICKUP_DELAY
    rework_count = 0

    #print(f'Order received with {sum(burger.values()):<2} ingredients: {burger}')
    while True:
        prep_wait_start = env.now
        with burgenerator.prep.request() as req:
            yield req
            prep_wait_end = env.now
            burgenerator.timeline_events.append({
                'order_id': order_id,
                'stage': 'prep_wait',
                'start': prep_wait_start,
                'end': prep_wait_end,
                'resource': 'prep_station'
            })
            burgenerator.usage_stats['prep_wait'].append(prep_wait_end - prep_wait_start)

            prep_start = env.now
            prep_duration = yield env.process(burgenerator.prepare())
            prep_end = env.now
            burgenerator.timeline_events.append({
                'order_id': order_id,
                'stage': 'prep',
                'start': prep_start,
                'end': prep_end,
                'resource': 'prep_station'
            })
            burgenerator.usage_stats['prep_work'].append(prep_duration)

        # if the pickup time is more than 5 minutes in the future don't start the burger
        if env.now < pickup_time - 300:
            assembly_wait_to_begin_start = env.now
            dt = (pickup_time - 300) - env.now
            burgenerator.usage_stats['assembly_wait_to_begin'].append(dt)
            yield env.timeout(dt)
            assembly_wait_to_begin_end = env.now
            burgenerator.timeline_events.append({
                'order_id': order_id,
                'stage': 'assembly_wait_to_begin',
                'start': assembly_wait_to_begin_start,
                'end': assembly_wait_to_begin_end,
                'resource': 'assembly'
            })

        assembly_wait_start = env.now
        with burgenerator.assembly.request() as req:
            yield req
            assembly_wait_end = env.now
            burgenerator.timeline_events.append({
                'order_id': order_id,
                'stage': 'assembly_wait',
                'start': assembly_wait_start,
                'end': assembly_wait_end,
                'resource': 'assembly'
            })
            burgenerator.usage_stats['assembly_wait'].append(assembly_wait_end - assembly_wait_start)

            assembly_start = env.now
            assembly_duration = yield env.process(burgenerator.assemble(burger))
            assembly_end = env.now
            burgenerator.timeline_events.append({
                'order_id': order_id,
                'stage': 'assembly_work',
                'start': assembly_start,
                'end': assembly_end,
                'resource': 'assembly'
            })
            burgenerator.usage_stats['assembly_work'].append(assembly_duration)

            # after assembling, errors can be found with a 5% chance
            if GEN.random() < c.FAIL_PROB:
                rework_count += 1
                # if burger assembly fails re-enter the loop once more
                # such that the failed burger gets remade immediately
                continue
            else:
                packing_start = env.now
                packing_duration = yield env.process(burgenerator.package())
                packing_end = env.now
                burgenerator.timeline_events.append({
                    'order_id': order_id,
                    'stage': 'assembly_packing',
                    'start': packing_start,
                    'end': packing_end,
                    'resource': 'assembly'
                })
                burgenerator.usage_stats['assembly_work'].append(packing_duration)
                # if the burger is assembled successfully we can break out of the loop
                break

    burgenerator.order_stats['total_time'].append(env.now - order_time)
    burgenerator.order_stats['num_reworks'].append(rework_count)

def _analyze_results(burgenerator):
    """Collects various statistics for a single simulation run."""
    stats = {}
    total_time = burgenerator.env.now

    stats['avg_prep_wait'] = np.mean(burgenerator.usage_stats['prep_wait'])
    stats['avg_assembly_wait'] = np.mean(burgenerator.usage_stats['assembly_wait'])
    stats['total_sim_duration'] = total_time
    stats['avg_process_time_per_burger'] =  np.mean(burgenerator.order_stats['total_time'])
    total_prep_work = np.sum(burgenerator.usage_stats['prep_work'])
    total_assembly_work = np.sum(burgenerator.usage_stats['assembly_work'])
    total_prep_occupied = total_prep_work
    total_prep_time_available = c.N_LINECOOKS * total_time
    total_assembly_time_available = c.N_ASSEMBLERS * total_time
    stats['prep_idle_percent'] = (1 - (total_prep_occupied / total_prep_time_available)) * 100
    stats['assembly_idle_percent'] = (1 - (total_assembly_work / total_assembly_time_available)) * 100

    return stats

def run_single_simulation(seed):
    """Runs a single simulation with a set seed. Seed generation is random each time the method is called."""
    np.random.seed(seed)
    global GEN
    GEN = np.random.default_rng(seed)
    env = simpy.Environment()
    burgenerator = Burgenerator(env)

    env.process(_incoming_orders(env, burgenerator))
    env.run()

    return burgenerator

def log_print(*args, **kwargs):
    """Logs and simultaneously prints the given arguments to console. This was written by Gemini CLI"""
    message = ' '.join(str(a) for a in args)
    with open('simlog.txt', 'a') as f:
        f.write(message + ('\n' if not kwargs.get('end') else ''))
    print(*args, **kwargs)

def run_simulations():
    """Orchestrates N_SIMS simulations, collects their statistics and calculates measures to answer the questions of the
    task. Printing strings and their formatting have been contributed by Gemini CLI."""
    log_print(f'Running {c.N_SIMS} simulations using {c.N_LINECOOKS} linecooks and {c.N_ASSEMBLERS} assemblers.')
    all_results = []
    for i in range(c.N_SIMS):
        seed = c.SEED + i
        burgenerator = run_single_simulation(seed)
        all_results.append(_analyze_results(burgenerator))

    df = pd.DataFrame(all_results)

    log_print('\nQA1: Does the burgermaster have waiting times due to missing ingredients?')
    log_print('\nQA2: How much unused ingredients are there at the end of the day?')

    log_print(f'\nQB: How does increasing/decreasing the security threshold affect the waiting times of the burgermaster?')
    log_print(f'QB1: Increase to 17.5%:')
    log_print(f'QB2: Increase to 20%:')
    log_print(f'QB3: Decrease to 12,5%:')
    log_print(f'QB4: Decrease to 10%:')

    log_print('\nQC: How do waiting times and unused ingredients at the end of the day change if the capacity of the'
              ' bins changes?')
    log_print(f'QC1: Waiting times with 60 items capacity:')
    log_print(f'QC1: Unused ingredients with 60 items capacity:')
    log_print(f'QC2: Waiting times with 90 items capacity:')
    log_print(f'QC2: Unused ingredients with 90 items capacity:')

    log_print('-' * 16, 'END OF RUN', '-' * 16, '\n')

if __name__ == '__main__':
    run_simulations()