"""
Planning and Configuration: Practical Task 1

Scenario:
  Students can order a burger online. The warm ingredients for this burger are then prepared by the 'Vorarbeitstheke'.
  The 'Burgermeister' assembles the warm and cold ingredients at the 'Zubereitungstheke' and puts the finished burger
  into the 'Auslage' where the burger waits to be picked up. Workers are considered resources.
  Burger orders are artificially created based on previously derived distributions (see data_analysis.ipynb).

Implementation Details:
  This implementation is mostly based on the CarWash example in simpy's documentation. To get a quick skeleton of the
  simulation pipeline Google's Gemini CLI has been used to propose an outline and code scaffolding that then could be
  filled in with implementation details by myself using the simpy documentation as reference. I got stuck on the
  implementation of the 5% fail chance. Here came another notable contribution of Gemini CLI with the idea of the
  'while True:' section in the _burger_process method to elegantly implement the breakout condition of a failed burger.

  Burger sampling is done with the derived distributions but could have been abstracted further as it is not of interest
  which cold ingredients get chosen just the total amount.

  The method _incoming_orders handles the orders that are submitted during the valid order window. This is the main
  driver of the simulation. Once an order has been generated it gets processed by the _burger_process method which
  queues the preparation and assembly steps according to the simulation details given in the task.

  Gemini CLI has also been used to implement an elegant logging solution of simulation statistics during the simulation
  runs, since my experience with these datastructures is limited. The benefit was a very fast development of the
  otherwise tedious overhead of logging such that I was able to focus my time on the actual simulation implementation.

Insights:
  With 2 linecooks and 1 burgermaster the burgenerator has a significant bottleneck at the prep station (linecooks) as
  they have basically 0 idle time and are constantly working (1.12% idle time). The burgermaster on the other hand works
  only ~40% of the time as they need to wait for the prep station. At this rate with 30-40 incoming orders per hour it
  takes more than 7.5 hours to clear all orders. The average wait time for a burger exceeds the pickup time by a delta
  of ~115 minutes. The burger needs ~145 minutes of processing time counted from receiving the order until it has been
  packed.

  Increasing the number of linecooks (i.e. relieving the bottleneck) quickly reduces the total lunchtime to ~4 hours
  at 4 linecooks. With 4 linecooks students only wait around 7.5 minutes for their burger after their original pickup
  window. Increasing the burgermasters (assemblers) has no meaningful impact at 4 linecooks. This is also supported by
  the comparably low idle time for assemblers of ~24% (1 assembler, 4 linecooks) versus ~60% (2 assemblers, 4 linecooks).
  The most even distribution of idle time was achieved by 5 linecooks and 1 assembler minimizing both idletimes to about
  14% for both stations. The resource utilization visualizations also show this effect pretty well as the utilization
  bars are almost fully filled out at a 5 to 1 ratio.
"""
import random
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
    stats['avg_student_wait'] = np.mean(burgenerator.order_stats['total_time']) - c.PICKUP_DELAY # we have to deduct the pickup delay here to get the true wait time on a burger

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

    avg_prep_wait = df['avg_prep_wait'].mean()
    avg_assembly_wait = df['avg_assembly_wait'].mean()
    bottleneck = 'Prep Station (linecooks)' if avg_prep_wait > avg_assembly_wait else 'Assembly Station (burgermaster)'
    log_print('\nQ1: Which areas of work lead to bottlenecks?')
    log_print(f'- An avg. order waits {avg_prep_wait:.2f} seconds to be prepared')
    log_print(f'- An avg. order waits {avg_assembly_wait:.2f} seconds to be assembled')
    log_print(f'--> The primary bottleneck is the {bottleneck}.')

    avg_lunch_break = df['total_sim_duration'].mean()
    log_print(f'\nQ2: How long does the average lunch break last at counter 3?')
    log_print(f'The average time to clear all orders is {avg_lunch_break / 3600:.2f} hours ({avg_lunch_break:.2f} seconds).')

    avg_prep_idle = df['prep_idle_percent'].mean()
    avg_assembly_idle = df['assembly_idle_percent'].mean()
    log_print('\nQ3: How much idle time do the linecooks and the Burgermaster have on average?')
    log_print(f'- Linecooks (Prep): {avg_prep_idle:.2f}% idle time.')
    log_print(f'- Burgermaster (Assembly): {avg_assembly_idle:.2f}% idle time.')

    avg_student_wait = df['avg_student_wait'].mean()
    log_print(f'\nQ4: How long do students wait on average for their burgers?')
    log_print(f'Students wait on average {avg_student_wait / 60:.2f} minutes ({avg_student_wait:.2f} seconds).')

    avg_process_time_per_burger = df['avg_process_time_per_burger'].mean()
    log_print(f'\nThe average burger takes {avg_process_time_per_burger / 60:.2f} minutes '
              f'({avg_process_time_per_burger:.2f} seconds) to process from receiving the order to finish packaging.')
    log_print('-' * 16, 'END OF RUN', '-' * 16, '\n')

if __name__ == '__main__':
    run_simulations()