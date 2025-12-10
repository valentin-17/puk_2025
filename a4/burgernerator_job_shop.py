"""
Planning and Configuration: Practical Task 3

Scenario:
  Instead of looking at the burger orders and the individual workers that process the orders this task proposed a
  job-shop configuration with a focus on the flow of the burger order through different assembly stations. For this each
  machine group has been initialized as a simpy resource with one worker.

Implementation Details:
  This implementation was written mostly from scratch since the old code from task 1 and 2 did not fit the new structure.
  What remained similar is the plotting and analysis code that has been adapted to fit the task at hand. This adaptation
  of plotting and analysis code (tracking of stats, accumulating data into dataframes etc.) has been done by ChatGPT in
  large parts.

  The core idea of this implementation is that the machines in the workshop are simpy resources with one worker each.
  These resources are chained depending on the burger order which is passed through the individual stages of assembly.
  All four burgertypes are assumed to occur uniformly distributed. There is no implementation of failure because it would
  add a lot of complexity since we'd have to think of when to put the burger back into the queue and model this. This
  could also lead to difficulties in task 4 which should build on this implementation.

  ChatGPT also helped a lot in implementing some methods like the generic _sample_processing_time method or the timing
  calculations within job_process.
"""
from collections import defaultdict
from typing import List, Dict, Any, Literal
import numpy as np
import simpy
import pandas as pd

import a3.config as c

GEN = np.random.default_rng(c.SEED)

def compute_theoretical_min_durations() -> Dict[str, float]:
    """Return the theoretical minimum throughput time (sum of expected process times) per burger type."""
    def expected_time(machine: str, variant: str) -> float:
        dist_tuple = c.PROCESSING_TIME[machine][variant]
        dist = dist_tuple[0].lower()
        if dist == 'normal':
            _, mu, _ = dist_tuple
            return float(mu)
        if dist == 'exponential':
            _, mean = dist_tuple
            return float(mean)
        if dist == 'uniform':
            _, low, high = dist_tuple
            return float((low + high) / 2)
        return 0.0

    return {
        burger: sum(expected_time(machine, variant) for machine, variant in route)
        for burger, route in c.ROUTINGS.items()
    }


class BurgeneratorJobShop:
    """Burgenerator workshop model with 5 machine groups and routing per burger type. Tracks timeline events, per-stage
     wait and work durations, order stats and resource busy time for utilization calculation."""
    def __init__(self, env: simpy.Environment,
                 queue_rule: Literal["FIFO", "LIFO", "SPT", "LPT", "RANDOM", "EDD"] = "FIFO"):
        self.env = env
        self.queue_rule = queue_rule  # "FIFO", "LIFO", "SPT", "LPT", "RANDOM", "EDD"

        # PriorityResource
        self.machines: Dict[str, simpy.PriorityResource] = {
            f'M{i+1}': simpy.PriorityResource(env, capacity=1)
            for i in range(5)
        }

        self._priority_counter = 0  # for FIFO/LIFO counting
        self.usage_stats: defaultdict[str, list] = defaultdict(list)
        self.order_stats: defaultdict[str, list] = defaultdict(list)
        self.timeline_events: List[Dict[str, Any]] = []

        self._machine_busy_periods = []
        self.routing = c.ROUTINGS
        self.processing_time = c.PROCESSING_TIME
        self.order_due_dates: Dict[int, float] = {}


    def _sample_processing_time(self, machine: str, variant: str) -> float | None:
        dist_tuple = self.processing_time[machine][variant]

        dist = dist_tuple[0].lower()
        if dist == 'normal':
            _, mu, sigma = dist_tuple
            return max(0.0, float(GEN.normal(mu, sigma)))
        if dist == 'exponential':
            _, mean = dist_tuple
            return float(GEN.exponential(mean))
        if dist == 'uniform':
            _, low, high = dist_tuple
            return float(GEN.uniform(low, high))
        return None

    def _expected_processing_time(self, machine: str, variant: str) -> float:
        """Return the mean processing time for a machine/variant combination."""
        dist_tuple = self.processing_time[machine][variant]
        dist = dist_tuple[0].lower()
        if dist == 'normal':
            _, mu, _ = dist_tuple
            return float(mu)
        if dist == 'exponential':
            _, mean = dist_tuple
            return float(mean)
        if dist == 'uniform':
            _, low, high = dist_tuple
            return float((low + high) / 2)
        return 0.0

    def _compute_due_date(self, burger_type: str, arrival_time: float) -> float:
        """Estimate an absolute due date based on arrival plus expected processing of the full routing."""
        expected_total = sum(
            self._expected_processing_time(machine, variant)
            for machine, variant in self.routing[burger_type]
        )
        return arrival_time + expected_total

    def _get_priority(self, machine: str, variant: str, proc_time: float, due_date: float | None = None) -> float:
        """
        Map the configured queue_rule (FIFO, LIFO, SPT, LPT, RANDOM, EDD)
        to a numeric priority for simpy.PriorityResource.
        Lower numbers = higher priority.
        """
        # FIFO: increasing counter → earlier arrivals get higher priority
        if self.queue_rule == "FIFO":
            self._priority_counter += 1
            return float(self._priority_counter)

        # LIFO: decreasing counter → most recent arrival gets highest priority
        if self.queue_rule == "LIFO":
            self._priority_counter -= 1
            return float(self._priority_counter)

        # SPT: shortest processing time first
        if self.queue_rule == "SPT":
            return float(proc_time)  # smaller time = higher priority

        # LPT: longest processing time first
        if self.queue_rule == "LPT":
            return float(-proc_time)  # larger time -> more negative -> higher priority

        # RANDOM: random priority
        if self.queue_rule == "RANDOM":
            return float(GEN.random())

        # EDD: earliest due date first (absolute due dates)
        if self.queue_rule == "EDD":
            self._priority_counter += 1  # keep deterministic tie-breaking
            if due_date is None:
                return float(self._priority_counter)
            return float(due_date + self._priority_counter * 1e-6)

        # Fallback = FIFO
        self._priority_counter += 1
        return float(self._priority_counter)


    # wrapper to request a machine and track busy-periods
    def _use_machine(self, machine: str, variant: str, order_id: int, due_date: float | None):
        """Return a process that requests the machine, performs processing and records stats."""
        res = self.machines[machine]

        # Sample processing time
        proc_time = self._sample_processing_time(machine, variant)

        # priorty based one queuing strategy
        priority = self._get_priority(machine, variant, proc_time, due_date)

        # from queue with highest priorty
        req = res.request(priority=priority)
        wait_start = self.env.now
        yield req
        wait_end = self.env.now

        # record waiting time
        self.usage_stats[f'{machine}_wait'].append(wait_end - wait_start)
        self.timeline_events.append({
            'time': self.env.now,
            'stage': f'{machine}_wait_end',
            'start': wait_start,
            'end': wait_end,
            'resource': machine,
            'variant': variant,
            'priority': priority,
            'order_id': order_id,
            'due_date': due_date,
        })

        # sample processing time and do work
        work_start = self.env.now
        # record busy period start
        busy_start = self.env.now
        yield self.env.timeout(proc_time)
        busy_end = self.env.now
        work_end = self.env.now

        # append busy period for utilization calc (one tuple per usage)
        self._machine_busy_periods.append((machine, busy_start, busy_end))

        # record work time & timeline
        self.usage_stats[f'{machine}_work'].append(proc_time)
        self.timeline_events.append({
            'time': self.env.now,
            'stage': f'{machine}_work',
            'start': work_start,
            'end': work_end,
            'resource': machine,
            'variant': variant,
            'duration': proc_time,
            'priority': priority,
            'order_id': order_id,
            'due_date': due_date,
        })

        res.release(req)
        return proc_time

def job_process(env: simpy.Environment, shop: BurgeneratorJobShop, burger_type: str, order_id: int):
    """Simulate an individual order passing through the routing for its burger_type. Collects per-order stats in
    shop.order_stats and timeline events."""
    arrival = env.now
    due_date = shop._compute_due_date(burger_type, arrival)
    shop.order_due_dates[order_id] = due_date
    shop.order_stats['due_date'].append(due_date)
    shop.timeline_events.append({
        'order_id': order_id,
        'stage': 'arrival',
        'time': arrival,
        'burger_type': burger_type,
        'due_date': due_date,
    })

    # track per-job wait total and work total for easy summary
    job_wait = 0.0
    job_work = 0.0

    for (machine, variant) in shop.routing[burger_type]:
        wait_start = env.now
        proc = yield env.process(shop._use_machine(machine, variant, order_id, due_date))
        # compute waiting time = time until work start
        last_ev = next(
            ev for ev in reversed(shop.timeline_events)
            if ev['stage'] == f'{machine}_work'
        )
        work_start = last_ev['start']
        job_wait += max(0, work_start - wait_start)
        job_work += proc

    # job finished
    finish = env.now
    sojourn = finish - arrival
    shop.order_stats['sojourn_time'].append(sojourn)
    shop.order_stats['job_wait_time'].append(job_wait)
    shop.order_stats['job_work_time'].append(job_work)
    shop.order_stats['lateness'].append(finish - due_date)
    shop.order_stats['burger_type'].append(burger_type)

    shop.timeline_events.append({
        'order_id': order_id,
        'stage': 'finished',
        'arrival': arrival,
        'finish': finish,
        'sojourn': sojourn,
        'burger_type': burger_type,
        'due_date': due_date
    })

def arrival_generator(env: simpy.Environment, shop: BurgeneratorJobShop):
    """Generates jobs during the arrival window defined by c.SIM_START and c.SIM_END."""
    burger_types = list(shop.routing.keys())
    sim_end = c.SIM_END
    order_id = 0

    # schedule until env.now reaches sim_end
    while env.now < sim_end:
        # sample burger type uniformly
        burger = GEN.choice(burger_types)
        env.process(job_process(env, shop, burger, order_id))
        order_id += 1
        interarrival = max(0.0, float(GEN.exponential(c.ARRIVAL_MEAN)))

        # advance time (do not allow arrivals after sim_end)
        if env.now + interarrival > sim_end:
            # jump to sim_end and exit loop (no more arrivals)
            yield env.timeout(max(0.0, sim_end - env.now))
            break
        else:
            yield env.timeout(interarrival)


def analyze_results(shop: BurgeneratorJobShop) -> Dict[str, Any]:
    """Return summary statistics for the run."""
    stats: Dict[str, Any] = {}
    total_sim_time = shop.env.now
    stats['total_sim_duration'] = total_sim_time

    stats['order_stats'] = shop.order_stats
    stats['timeline_events'] = shop.timeline_events

    # sojourn times
    sojourns = np.array(shop.order_stats.get('sojourn_time', []))
    if sojourns.size > 0:
        stats['avg_sojourn'] = float(np.mean(sojourns))
        stats['median_sojourn'] = float(np.median(sojourns))
        stats['std_sojourn'] = float(np.std(sojourns, ddof=1)) if sojourns.size > 1 else 0.0
        stats['n_completed'] = int(sojourns.size)
    else:
        stats['avg_sojourn'] = stats['median_sojourn'] = stats['std_sojourn'] = 0.0
        stats['n_completed'] = 0

    # per-machine utilization: sum busy times for machine / total_sim_time
    machine_names = list(shop.machines.keys())

    for m in machine_names:
        stats[f'{m}_utilization'] = 0.0
        stats[f'{m}_avg_wait'] = np.mean(shop.usage_stats.get(f'{m}_wait', [0.0])) if shop.usage_stats.get(f'{m}_wait') else 0.0
        stats[f'{m}_avg_work'] = np.mean(shop.usage_stats.get(f'{m}_work', [0.0])) if shop.usage_stats.get(f'{m}_work') else 0.0

    # aggregate busy periods
    busy_by_machine = defaultdict(float)
    for (mname, start, end) in shop._machine_busy_periods:
        busy_by_machine[mname] += (end - start)

    for i, m in enumerate(machine_names):
        busy = busy_by_machine.get(m, 0.0)
        stats[f'{m}_busy_total'] = busy
        stats[f'{m}_utilization'] = (busy / total_sim_time) * 100.0 if total_sim_time > 0 else 0.0

    # overall bottleneck heuristic: machine with the highest utilization
    utilizations = {m: stats[f'{m}_utilization'] for m in machine_names}
    if utilizations:
        bottleneck_machine = max(utilizations.items(), key=lambda kv: kv[1])[0]
        stats['bottleneck'] = bottleneck_machine
    else:
        stats['bottleneck'] = None

    # average job wait time and work time
    job_waits = np.array(shop.order_stats.get('job_wait_time', []))
    job_works = np.array(shop.order_stats.get('job_work_time', []))
    stats['avg_job_wait'] = float(np.mean(job_waits)) if job_waits.size > 0 else 0.0
    stats['avg_job_work'] = float(np.mean(job_works)) if job_works.size > 0 else 0.0

    # per-burger-type sojourn times
    types = shop.order_stats.get('burger_type', [])
    per_type_sojourn: Dict[str, float] = {}
    if types:
        df_jobs = pd.DataFrame({
            'burger_type': types,
            'sojourn': shop.order_stats.get('sojourn_time', []),
        })
        per_group = df_jobs.groupby('burger_type')['sojourn'].mean()
        per_type_sojourn = {bt: float(val) for bt, val in per_group.items()}
    stats['per_type_sojourn'] = per_type_sojourn

    # Variant-level stats
    rows = []
    for ev in shop.timeline_events:
        if '_work' in ev.get('stage', ''):
            rows.append({
                'machine': ev['resource'],
                'variant': ev.get('variant', ''),
                'duration': ev['duration'],
            })

    df_mv = pd.DataFrame(rows)
    if not df_mv.empty:
        mv_group = df_mv.groupby(['machine', 'variant'])
        machine_variant_table = []
        sim_time = shop.env.now

        for (m, v), grp in mv_group:
            total_busy = grp['duration'].sum()
            util = (total_busy / sim_time) * 100

            waits = shop.usage_stats.get(f'{m}_wait', [])
            works = shop.usage_stats.get(f'{m}_work', [])

            machine_variant_table.append({
                'machine': m,
                'variant': v,
                'util_percent': util,
                'avg_wait_s': np.mean(waits) if waits else 0.0,
                'avg_work_s': np.mean(works) if works else 0.0,
                'total_busy_s': total_busy,
            })

        stats['machine_variant_table'] = pd.DataFrame(machine_variant_table)
    else:
        stats['machine_variant_table'] = pd.DataFrame([])

    return stats


def run_single_simulation(seed: int = None, queue_rule: Literal["FIFO", "LIFO", "SPT", "LPT", "RANDOM", "EDD"] = "FIFO") -> BurgeneratorJobShop:
    """Run a single simulation."""
    np.random.seed(seed)
    global GEN
    GEN = np.random.default_rng(seed)

    env = simpy.Environment()
    shop = BurgeneratorJobShop(env, queue_rule=queue_rule)

    env.process(arrival_generator(env, shop))
    env.run()  # run until no events left (arrivals stop at SIM_END by generator)
    return shop


def log_print(*args, **kwargs):
    message = ' '.join(str(a) for a in args)
    with open('simlog.txt', 'a') as f:
        f.write(message + ('\n' if not kwargs.get('end') else ''))
    print(*args, **kwargs)


def run_simulations(queue_rule: Literal["FIFO", "LIFO", "SPT", "LPT", "RANDOM", "EDD"] = "FIFO",
                    compare_all_rules: bool = True):
    """Orchestrate multiple simulations, compute aggregated metrics and print them.

    If compare_all_rules is True, runs the requested analysis for FIFO, LIFO, SPT, LPT and RANDOM and logs the
    requested comparisons (theoretical vs. actual per burger type, plus a recommendation).
    """
    n_sims = c.N_SIMS
    queue_rules = ["FIFO", "LIFO", "SPT", "LPT", "RANDOM"] if compare_all_rules else [queue_rule]

    log_print(f'Running {n_sims} replications of the job-shop model.')

    theoretical_min = compute_theoretical_min_durations()
    log_print("\nTheoretical minimal sojourn time (s) per burger type:")
    for burger, t in theoretical_min.items():
        log_print(f"- {burger}: {t:.2f} s")

    overall_sojourn_by_rule: Dict[str, float] = {}

    for rule in queue_rules:
        log_print(f'\n--- Strategy {rule} ---')
        all_stats = []
        for i in range(n_sims):
            seed = c.SEED + i
            shop = run_single_simulation(seed, queue_rule=rule)
            stats = analyze_results(shop)
            all_stats.append(stats)

        # convert some selected scalar outputs into DataFrame for averaging
        df = pd.DataFrame([{
            'avg_sojourn': s['avg_sojourn'],
            'n_completed': s['n_completed'],
            'avg_job_wait': s['avg_job_wait'],
            'avg_job_work': s['avg_job_work'],
            **{m: s.get(f'{m}_utilization', 0.0) for m in list(all_stats[0]['machine_variant_table']['machine'])}
        } for s in all_stats])

        # Bottlenecks
        avg_utils = df[[col for col in df.columns if col.startswith('M')]].mean()
        top_bottleneck = avg_utils.idxmax()
        log_print('Q1: Which areas of work lead to bottlenecks?')
        for m, util in avg_utils.items():
            log_print(f'- {m}: mean utilization {util:.2f}%')
        log_print(f'--> Primary bottleneck (by mean util): {top_bottleneck}')

        # idle times per machine (100 - utilization)
        log_print('Q3: Idle times per machine (avg):')
        for m, util in avg_utils.items():
            log_print(f'- {m}: {(100.0 - util):.2f}% idle')

        log_print("Queue rule: ", rule)
        log_print('Summary: average sojourn time:')
        log_print(f"- {df['avg_sojourn'].mean()/60.0:.2f} minutes ({df['avg_sojourn'].mean():.2f} s)")
        overall_sojourn_by_rule[rule] = float(df['avg_sojourn'].mean())

        # per burger type comparison to theoretical minima
        per_type_actual: Dict[str, float] = defaultdict(float)
        per_type_counts: Dict[str, int] = defaultdict(int)
        for stat in all_stats:
            for bt, val in stat.get('per_type_sojourn', {}).items():
                per_type_actual[bt] += val
                per_type_counts[bt] += 1

        log_print("Sojourn time per machine (mean of runs) and difference of minimum:")
        for bt, min_val in theoretical_min.items():
            actual_mean = per_type_actual[bt] / per_type_counts[bt] if per_type_counts[bt] else 0.0
            diff = actual_mean - min_val
            log_print(f"- {bt}: Actual {actual_mean:.2f} s | Minimum {min_val:.2f} s | Difference {diff:.2f} s")


    log_print('-' * 40, 'END OF RUN', '-' * 40)
    return all_stats


if __name__ == '__main__':
    run_simulations()