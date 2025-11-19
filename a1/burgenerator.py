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
  implementation of the 5% fail chance. Here came another notable contribution of Gemini CLI with the 'while True:'
  section in the _burger_process method to elegantly implement the breakout condition of a failed burger.

  Burger sampling is done with the derived distributions but could have been abstracted further as it is not of interest
  which cold ingredients get chosen just the total amount.

  The method _incoming_orders handles the orders that are submitted during the valid order window. This is the main
  driver of the simulation. Once an order has been generated it gets processed by the _burger_process method which
  queues the preparation and assembly steps according to the simulation details given in the task.

  Gemini CLI has also been used to implement an elegant logging solution of simulation statistics during the simulation
  runs, since my experience with these datastructures is limited. The benefit was a very fast development of the
  otherwise tedious overhead of logging such that I was able to focus my time on the actual simulation implementation.
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

    def prepare(self):
        # we can assume that the prep_time is always freezer prep time and warm prep time since every burger has to have
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
    while env.now < order_window:
        burger = _sample_burger()
        env.process(_burger_process(env, burgenerator, burger))
        order_downtime = int(GEN.normal(c.TIME_BETWEEN_ARRIVALS_MEAN, c.TIME_BETWEEN_ARRIVALS_SCALE))

        yield env.timeout(max(0, order_downtime))

def _sample_burger():
    """
    Samples a random burger with MIN_INGREDIENTS and MAX_INGREDIENTS and at least one bun and one patty. Invalid burgers
    are immediately rejected.
    """
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

def _burger_process(env, burgenerator, burger):
    order_time = env.now
    pickup_time = order_time + c.PICKUP_DELAY
    rework_count = 0

    #print(f'Order received with {sum(burger.values()):<2} ingredients: {burger}')
    while True:
        prep_wait_start = env.now
        with burgenerator.prep.request() as req:
            yield req
            burgenerator.usage_stats['prep_wait'].append(env.now - prep_wait_start)
            prep_duration = yield env.process(burgenerator.prepare())
            burgenerator.usage_stats['prep_work'].append(prep_duration)

        # if the pickup time is more than 5 minutes in the future don't start the burger
        if env.now < pickup_time - 300:
            dt = (pickup_time - 300) - env.now
            burgenerator.usage_stats['assembly_wait_to_begin'].append(dt)
            yield env.timeout(dt)

        assembly_wait_start = env.now
        with burgenerator.assembly.request() as req:
            yield req
            burgenerator.usage_stats['assembly_wait'].append(env.now - assembly_wait_start)
            assembly_duration = yield env.process(burgenerator.assemble(burger))
            burgenerator.usage_stats['assembly_work'].append(assembly_duration)

            # after assembling, errors can be found with a 5% chance
            if GEN.random() < c.FAIL_PROB:
                rework_count += 1
                # if burger assembly fails re-enter the loop once more
                # such that the failed burger gets remade immediately
                continue
            else:
                packing_duration = yield env.process(burgenerator.package())
                burgenerator.usage_stats['assembly_work'].append(packing_duration)
                # if the burger is assembled successfully we can break out of the loop
                break

    burgenerator.order_stats['total_time'].append(env.now - order_time)
    burgenerator.order_stats['num_reworks'].append(rework_count)

def _analyze_results(burgenerator):
    stats = {}
    total_time = burgenerator.env.now

    stats['avg_prep_wait'] = np.mean(burgenerator.usage_stats['prep_wait'])
    stats['avg_assembly_wait'] = np.mean(burgenerator.usage_stats['assembly_wait'])
    stats['total_sim_duration'] = total_time
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
    np.random.seed(seed)
    global GEN
    GEN = np.random.default_rng(seed)
    env = simpy.Environment()
    burgenerator = Burgenerator(env)

    env.process(_incoming_orders(env, burgenerator))
    env.run()

    return _analyze_results(burgenerator)

def run_simulations():
    print(f"Running {c.N_SIMS} simulations...")
    all_results = []
    for i in range(c.N_SIMS):
        seed = c.SEED + i
        all_results.append(run_single_simulation(seed))
        print(f"  Simulation {i + 1}/{c.N_SIMS} complete.",end='\r')
    print("\n\n--- Simulation Analysis Complete ---")

    df = pd.DataFrame(all_results)

    avg_prep_wait = df['avg_prep_wait'].mean()
    avg_assembly_wait = df['avg_assembly_wait'].mean()
    bottleneck = "Prep Station (Line Cooks)" if avg_prep_wait > avg_assembly_wait else "Assembly Station (Burger Chef)"
    print("\n1. Which areas of work lead to bottlenecks?")
    print(f"- Avg. Wait for Prep: {avg_prep_wait:.2f} seconds")
    print(f"- Avg. Wait for Assembly: {avg_assembly_wait:.2f} seconds")
    print(f"--> The primary bottleneck is the {bottleneck}.")

    avg_lunch_break = df['total_sim_duration'].mean()
    print(f"\n2. How long does the average lunch break last at counter 3?")
    print(f"The average time to clear all orders is {avg_lunch_break / 3600:.2f} hours ({avg_lunch_break:.2f} seconds).")

    avg_prep_idle = df['prep_idle_percent'].mean()
    avg_assembly_idle = df['assembly_idle_percent'].mean()
    print("\n3. How much idle time do the assistants and the burger chef have on average?")
    print(f"- Assistants (Prep): {avg_prep_idle:.2f}% idle time.")
    print(f"- Burger Chef (Assembly): {avg_assembly_idle:.2f}% idle time.")

    avg_student_wait = df['avg_student_wait'].mean()
    print(f"\n4. How long do students wait on average for their burgers?")
    print(f"Students wait on average {avg_student_wait / 60:.2f} minutes ({avg_student_wait:.2f} seconds).")
if __name__ == '__main__':
    run_simulations()