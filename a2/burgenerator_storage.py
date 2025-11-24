"""
Planning and Configuration: Practical Task 2

Scenario:
  In addition to the scenario from task 1 here the storage has to be modeled as well. For this a new resource 'helper'
  is introduced. The helper can fetch new ingredients from the infinite storage to refill the finite storage bins of the
  prep station as well as the assembly station. The helper can only carry one ingredient per trip.

Implementation Details:
  This implementation is an extension of the implementation from task 1. Helpers are implemented as simpy.Resource while
  the ingredient containers are modeled as simpy.Container. This allows the retrieving and adding from and to containers
  by multiple resources (burgermaster/helpers).

  To monitor the container fill level and effectively implement the s,Q policy a monitor method has been implemented that
  periodically (every 10th step) checks the fill level of each container against the defined reorder threshold. Once the
  reorder threshold of a container is reached the helper gets blocked to fetch the ingredients. During this time the
  container remains free such that the burgermaster or linecooks can continue working on their orders. This is realistic
  as they 'inform' the helper about a deminishing ingredient but as they have the reorder threshold as buffer they do not
  immediately run out of ingredients.

  This was kind of tricky to model the way I did here because I needed to guarantee that the monitor method knows when a
  reordering has been called to not call a reordering in the next step. This has been solved by using a set
  'pending_refills' that locks a container when it is in the process of being refilled. It is only locked for the request
  of a refill. For other activities it remains 'free'.

  Again Gemini CLI was used to assist in implementing this task. Especially all the logging and data collection have been
  heavily supported by the coding agent. Here I also faced a lot more problems with the coding agent in comparison to task
  1 as
"""
from collections import defaultdict
import subprocess
import numpy as np
import pandas as pd
import simpy

import a2.config as c


GEN = np.random.default_rng(c.SEED)


class Burgenerator:
    """Holds the metadata for Theke 3 during a simulation run."""
    def __init__(self, env: simpy.Environment, container_size=c.CONTAINER_SIZE):
        self.sim_end = None
        self.env = env
        self.prep: simpy.Resource = simpy.Resource(env, capacity=c.N_LINECOOKS)
        self.assembly: simpy.Resource = simpy.Resource(env, capacity=c.N_ASSEMBLERS)
        self.helper: simpy.Resource = simpy.Resource(env, capacity=c.N_HELPERS)
        self.container: dict[str, simpy.Container] = {container: simpy.Container(
            env, capacity=container_size, init=container_size) for container in c.CONTAINERS}
        self.pending_refill = set()
        self.usage_stats: defaultdict[str, list] = defaultdict(list)
        self.order_stats: defaultdict[str, list] = defaultdict(list)
        self.timeline_events = []
        self.container_levels = []


    def prepare(self, order_id, burger: dict[str, int]):
        # we can assume that the prep_time is only freezer prep time since the linecook takes the ingredients from the
        # freezer, puts them on the grill and then can move on to the next order. for simplicity’s sake we assume that
        # the grill always has enough space for the next patty. the prep time is then independent of the total amount of
        # warm ingredients.
        for warm_ingredient in c.WARM_INGREDIENTS:
            amount = burger.get(warm_ingredient)
            if amount < 1:
                continue
            
            start_wait = self.env.now
            yield self.container[warm_ingredient].get(amount)
            end_wait = self.env.now
            wait_time = end_wait - start_wait
            self.usage_stats['prep_wait_for_refill'].append(wait_time)
            self.timeline_events.append({
                'order_id': order_id,
                'stage': 'prep_wait_for_refill',
                'start': start_wait,
                'end': end_wait,
                'resource': 'prep'
            })

        prep_time = GEN.gamma(c.FREEZER_PREP_TIME_SHAPE, c.FREEZER_PREP_TIME_SCALE)
        yield self.env.timeout(prep_time)
        grill_event = self.env.process(self.grill())
        return prep_time, grill_event

    def grill(self):
        grill_time = GEN.uniform(c.WARM_PROCESSING_TIME_LOW, c.WARM_PROCESSING_TIME_HIGH)  # seconds
        yield self.env.timeout(grill_time)
        return grill_time

    def refill(self, name: str, amount: int):
        # lock the container so that it does not send extra calls to be refilled all the time
        self.pending_refill.add(name)
        with self.helper.request() as req:
            yield req

            refill_start = self.env.now
            refill_time = GEN.normal(c.REFILL_TIME_MEAN, c.REFILL_TIME_SCALE)
            yield self.env.timeout(refill_time)

            yield self.container[name].put(amount)
            refill_end = self.env.now
            self.timeline_events.append({
                'start': refill_start,
                'end': refill_end,
                'resource': 'helper'
            })
            self.pending_refill.discard(name)

            return refill_time

    def assemble(self, order_id, burger: dict[str, int], grill_event):
        # only count assembly time for the cold ingredients plus the toasting time, buns are also not considered a cold
        # ingredient as they have the 40-second toasting time
        yield grill_event

        for cold_ingredient in c.COLD_INGREDIENTS:
            amount = burger.get(cold_ingredient)
            if amount < 1:
                continue

            start_wait = self.env.now
            yield self.container[cold_ingredient].get(amount)
            end_wait = self.env.now
            wait_time = end_wait - start_wait
            self.usage_stats['assembly_wait_for_refill'].append(wait_time)
            self.timeline_events.append({
                'order_id': order_id,
                'stage': 'assembly_wait_for_refill',
                'start': start_wait,
                'end': end_wait,
                'resource': 'assembly'
            })

        assembly_time = (c.TOASTING_TIME
                         + (GEN.normal(c.ASSEMBLY_TIME_PER_INGREDIENT_MEAN, c.ASSEMBLY_TIME_PER_INGREDIENT_SCALE)
                            * sum(v for k, v in burger.items() if k not in ['bun', 'patty', 'bacon'])))
        yield self.env.timeout(assembly_time)
        return assembly_time

    def package(self):
        packing_time = GEN.uniform(c.PACKING_TIME_LOW, c.PACKING_TIME_HIGH)
        if GEN.random() < c.FRIES_PROB:
            packing_time += GEN.uniform(c.PACKING_TIME_FRIES_LOW, c.PACKING_TIME_FRIES_HIGH)
        yield self.env.timeout(packing_time)
        self.sim_end = self.env.now # log the latest package step as simulation end
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

def _container_monitor(env, burgenerator):
    while True:
        for name, container in burgenerator.container.items():
            burgenerator.container_levels.append({
                'time': env.now,
                'container': name,
                'level': container.level
            })
            
        reorder_threshold = c.CONTAINER_SIZE * c.REORDER_THRESHOLD

        for name in burgenerator.container:
            if name in burgenerator.pending_refill:
                continue

            container = burgenerator.container[name]

            if container.level < reorder_threshold:
                amount = container.capacity - container.level
                env.process(burgenerator.refill(name, amount))

        # checks for refill every 10th simulation step.
        yield env.timeout(10)

def _sample_burger() -> dict[str, int]:
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
    """Simulates an order process from incoming order to finished burger. This method uses the burgenerator class as the
    environment (i.e. Theke 3) in which the tasks are done. The individual tasks are modeled in the burgenerator class
    while inter-task activities like the failure of assembly or the wait to start the assembly are modeled here in this
    method. Further various timings and statistics are tracked to allow for later analysis."""
    while True:
        order_time = env.now
        pickup_time = order_time + c.PICKUP_DELAY
        rework_count = 0

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
                    'resource': 'prep'
                })
                burgenerator.usage_stats['prep_wait'].append(prep_wait_end - prep_wait_start)

                prep_start = env.now
                prep_duration, grill_event = yield env.process(burgenerator.prepare(order_id, burger))
                prep_end = env.now
                burgenerator.timeline_events.append({
                    'order_id': order_id,
                    'stage': 'prep_work',
                    'start': prep_start,
                    'end': prep_end,
                    'resource': 'prep'
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
                assembly_duration = yield env.process(burgenerator.assemble(order_id, burger, grill_event))
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
                    break

        burgenerator.order_stats['total_time'].append(env.now - order_time)
        burgenerator.order_stats['num_reworks'].append(rework_count)
        break

def _analyze_results(burgenerator):
    """Collects various statistics for a single simulation run."""
    stats = {}

    stats['total_sim_duration'] = burgenerator.sim_end
    stats['avg_process_time_per_burger'] = np.mean(burgenerator.order_stats['total_time'])
    stats['avg_assembly_wait_time_for_refill'] = np.mean(burgenerator.usage_stats['assembly_wait_for_refill'])

    leftover_ingredients = {name: container.level for name, container in burgenerator.container.items()}
    stats['leftover_ingredients'] = leftover_ingredients
    stats['total_leftover_ingredients'] = sum(leftover_ingredients.values())

    return stats

def run_single_simulation(seed, container_size=c.CONTAINER_SIZE, reorder_threshold=c.REORDER_THRESHOLD):
    """Runs a single simulation with a set seed."""
    np.random.seed(seed)
    global GEN
    GEN = np.random.default_rng(seed)

    original_reorder_threshold = c.REORDER_THRESHOLD
    c.REORDER_THRESHOLD = reorder_threshold
    
    env = simpy.Environment()
    burgenerator = Burgenerator(env, container_size=container_size)

    env.process(_container_monitor(env, burgenerator))
    env.process(_incoming_orders(env, burgenerator))
    env.run(until=3 * c.ORDER_TIME) # cuts the simulation after a set amount of time to prevent the resources getting stuck at the end

    c.REORDER_THRESHOLD = original_reorder_threshold
    
    return burgenerator

def log_print(*args, **kwargs):
    """Logs and simultaneously prints the given arguments to console."""
    message = ' '.join(str(a) for a in args)
    with open('simlog.txt', 'a') as f:
        f.write(message + ('\n' if not kwargs.get('end') else ''))
    print(*args, **kwargs)

def run_and_analyze_scenario(container_size=c.CONTAINER_SIZE, reorder_threshold=c.REORDER_THRESHOLD, return_levels=False):
    """Runs a set of simulations for a given scenario and returns aggregated results."""
    all_results = []
    container_levels_for_first_run = None
    for i in range(c.N_SIMS):
        seed = c.SEED + i
        burgenerator = run_single_simulation(seed, container_size, reorder_threshold)
        if i == 0 and return_levels:
            container_levels_for_first_run = burgenerator.container_levels
        all_results.append(_analyze_results(burgenerator))

    df = pd.DataFrame(all_results)

    aggregated_results = {
        'container_size': container_size,
        'avg_process_time_per_burger': df['avg_process_time_per_burger'].mean(),
        'avg_assembly_wait_time_for_refill': df['avg_assembly_wait_time_for_refill'].mean(),
        'avg_total_leftovers': df['total_leftover_ingredients'].mean(),
    }

    if return_levels:
        aggregated_results['container_levels'] = container_levels_for_first_run

    return aggregated_results


def run_and_collect_scenario_results(return_levels=False):
    """Runs all scenarios and collects their results."""
    all_scenario_results = []

    # Baseline
    baseline_results = run_and_analyze_scenario(reorder_threshold=c.REORDER_THRESHOLD, return_levels=return_levels)
    all_scenario_results.append({
        'name': 'Baseline',
        'results': baseline_results,
        'type': 'baseline'
    })

    # Thresholds
    thresholds = [0.10, 0.125, 0.175, 0.20]
    for thresh in thresholds:
        results = run_and_analyze_scenario(reorder_threshold=thresh, return_levels=return_levels)
        all_scenario_results.append({
            'name': f'Safety Stock {thresh:.1%}',
            'results': results,
            'type': 'threshold'
        })

    # Capacities
    capacities = [60, 90]
    for cap in capacities:
        results = run_and_analyze_scenario(container_size=cap, return_levels=return_levels)
        all_scenario_results.append({
            'name': f'Container Size {cap}',
            'results': results,
            'type': 'capacity'
        })

    return all_scenario_results

def run_simulations():
    """Orchestrates simulations for different scenarios and prints the analysis."""
    log_print(f'Running {c.N_SIMS} simulations using {c.N_LINECOOKS} linecooks, {c.N_ASSEMBLERS} assemblers and {c.N_HELPERS} helpers.')

    scenario_results = run_and_collect_scenario_results()

    baseline_scenario = next(item for item in scenario_results if item["type"] == "baseline")
    baseline_results = baseline_scenario['results']

    log_print('\nQA: Base scenario')
    log_print(f'    Safety Stock {c.SAFETY_STOCK_THRESHOLD:.1%}, capacity {c.CONTAINER_SIZE}:')
    log_print(f'      - Average assembly wait time for refill: {baseline_results['avg_assembly_wait_time_for_refill']:.2f} seconds.')
    log_print(f'      - Average process time: {baseline_results['avg_process_time_per_burger'] / 60:.2f} minutes ({baseline_results['avg_process_time_per_burger']:.2f} seconds).')
    log_print(f'      - Average leftover ingredients: {int(baseline_results['avg_total_leftovers'])} units.')

    log_print(f'\nQB: Various safety stocks, default container capacity ({c.CONTAINER_SIZE} units)')
    threshold_scenarios = [item for item in scenario_results if item['type'] == 'threshold']
    for scenario in threshold_scenarios:
        results = scenario['results']
        log_print(f'    {scenario['name']}:')
        log_print(f'      - Average assembly wait time for refill: {results['avg_assembly_wait_time_for_refill']:.2f} seconds.')
        log_print(f'      - Average process time: {results['avg_process_time_per_burger'] / 60:.2f} minutes ({results['avg_process_time_per_burger']:.2f} seconds).')
        log_print(f'      - Average leftover ingredients: {int(baseline_results['avg_total_leftovers'])} units.')

    log_print(f'\nQC: Various container capacities, default safety stock ({c.SAFETY_STOCK_THRESHOLD:.1%})')
    capacity_scenarios = [item for item in scenario_results if item['type'] == 'capacity']
    for scenario in capacity_scenarios:
        results = scenario['results']
        log_print(f'    {scenario['name']}:')
        log_print(f'      - Average assembly wait time for refill: {results['avg_assembly_wait_time_for_refill']:.2f} seconds.')
        log_print(f'      - Average process time: {results['avg_process_time_per_burger'] / 60:.2f} minutes ({results['avg_process_time_per_burger']:.2f} seconds).')
        log_print(f'      - Average leftover ingredients: {int(results['avg_total_leftovers'])} units.')

    log_print('\n' + '-' * 16, 'END OF RUN', '-' * 16, '\n')


if __name__ == '__main__':
    run_simulations()