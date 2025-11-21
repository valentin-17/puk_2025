import numpy as np
import simpy
from simpy import PriorityStore, FilterStore
from simpy.resources.store import FilterStoreGet, StorePut
from typing import (
    Any,
    NamedTuple,
    Optional,
)
from heapq import heappush
import matplotlib.pyplot as plt
import config as c

# -------------------------------------------------------------------
# Global settings
# -------------------------------------------------------------------
np.random.seed(42)      # set the seed
SIM_DURATION = 310      # simulation time


# -------------------------------------------------------------------
# Statistics + Gantt data
# -------------------------------------------------------------------
class Statistics:
    def __init__(self):
        # waiting times
        self.wait_before_zuarbeiter = []       # order -> taken by Zuarbeiter
        self.wait_before_burgermeister = []    # WarmSchrank entry -> taken by Burgermeister
        self.student_total_wait = []           # order -> final completion

        # idle times
        self.zuarbeiter_idle = {}              # name -> total idle time
        self.burgermeister_idle = 0.0

        # service times at Burgermeister (Theke 3)
        self.burgermeister_service_times = []  # per burger

    def add_zuarbeiter(self, name: str):
        if name not in self.zuarbeiter_idle:
            self.zuarbeiter_idle[name] = 0.0


stats = Statistics()

# Gantt data: one entry per successfully etikettierter burger
# each entry: {"id": burger_id, "start": order_time, "end": etikettieren_time}
burger_gantt_data = []


# -------------------------------------------------------------------
# Data structures
# -------------------------------------------------------------------
class DynamicDeadlineItem(NamedTuple):
    """Wrap a burger with a dynamic deadline."""
    burger: Any
    deadline: Any
    env: Any

    def get_burger(self):
        return self.burger

    def get_deadline(self):
        # remaining time until deadline
        return self.deadline - self.env.now

    def __lt__(self, other) -> bool:  # type: ignore[override]
        priority_1 = self.deadline - self.env.now
        priority_2 = other.deadline - other.env.now
        return priority_1 < priority_2


class PriorityFilterStore(FilterStore):
    """FilterStore that stores items in a heap based on their priority."""
    def _do_put(self, event: StorePut) -> Optional[bool]:
        if len(self.items) < self._capacity:
            heappush(self.items, event.item)
            event.succeed()
        return None


class BestellungList:
    """Order list with latency before entering the internal PriorityStore."""
    def __init__(self, env, delay):
        self.env = env
        self.delay = delay
        self.store = PriorityStore(env)

    def latency(self, value):
        yield self.env.timeout(self.delay)
        self.store.put(value)

    def put(self, value):
        self.env.process(self.latency(value))

    def get(self):
        return self.store.get()


class WarmSchrank:
    """Warm storage using PriorityFilterStore with latency before arrival."""
    def __init__(self, env, delay):
        self.env = env
        self.delay = delay
        self.store = PriorityFilterStore(env)

    def latency(self, value):
        yield self.env.timeout(self.delay)
        # mark time when the burger enters the WarmSchrank
        value.burger["entered_warmschrank"] = self.env.now
        self.store.put(value)

    def put(self, value):
        self.env.process(self.latency(value))

    def get(self, filter_func=None):
        return self.store.get(filter=filter_func)


# -------------------------------------------------------------------
# Processes
# -------------------------------------------------------------------
def Student(env, bestellung_list):
    """Students arrive, create burgers and place orders."""
    while env.now < 180:
        burger_dict = create_burger()
        burger_dict["order_time"] = env.now  # when the student orders

        burger_creation = DynamicDeadlineItem(
            burger_dict,
            deadline=env.now + 30,
            env=env
        )
        bestellung_list.put(burger_creation)
        print("bestellt at ", env.now)

        # inter-arrival time of students
        delay = np.random.normal(loc=105, scale=8) / 60
        yield env.timeout(delay)


def Zuarbeiter(env, bestellung_list, warm_schrank, name: str, stats: Statistics):
    """Helper worker that prepares burgers and puts them in the WarmSchrank."""
    stats.add_zuarbeiter(name)

    while True:
        # idle while waiting for an order
        idle_start = env.now
        bestellung = yield bestellung_list.get()
        stats.zuarbeiter_idle[name] += env.now - idle_start

        # waiting time before Zuarbeiter
        burger = bestellung.get_burger()
        if "order_time" in burger:
            stats.wait_before_zuarbeiter.append(env.now - burger["order_time"])

        # processing by helper
        delay = np.random.gamma(shape=10, scale=2) / 60
        yield env.timeout(delay)
        delay = np.random.uniform(low=360, high=600) / 60
        yield env.timeout(delay)

        warm_schrank.put(bestellung)


def Burgermeister(env, bestellung_list, warm_schrank, stats: Statistics):
    """Final station (Theke 3): etikettieren + finishing."""
    sum_deadline_violation = 0
    while True:
        # internal prep delay (not idle)
        delay = np.random.normal(loc=5, scale=1) / 60
        yield env.timeout(delay)

        # idle while waiting for warm burgers with soon deadline
        idle_start = env.now
        warme_zutaten = yield warm_schrank.get(lambda x: x.get_deadline() < 5)
        stats.burgermeister_idle += env.now - idle_start

        burger = warme_zutaten.get_burger()

        # waiting time in WarmSchrank before Burgermeister
        if "entered_warmschrank" in burger:
            stats.wait_before_burgermeister.append(env.now - burger["entered_warmschrank"])

        print("deadline lower 5: ", warme_zutaten.get_deadline())
        print(burger)

        service_start = env.now  # service time at Theke 3 starts here

        if np.random.rand() <= 0.05:
            print("Fehler der Zuberetiung")
            # burger must be re-done, put back into the order list
            bestellung_list.put(warme_zutaten)
        else:
            print("etikitieren zur deadline: ", warme_zutaten.get_deadline())

            # Etikettieren time is env.now (start of final wrapping)
            etikettieren_time = env.now

            # log Gantt data: from order -> etikettieren
            if "order_time" in burger:
                burger_gantt_data.append({
                    "id": burger["id"],
                    "start": burger["order_time"],
                    "end": etikettieren_time
                })

            # deadline violation statistics
            if warme_zutaten.get_deadline() < 0:
                sum_deadline_violation += warme_zutaten.get_deadline()

            # final processing steps
            delay = np.random.uniform(low=10, high=20) / 60
            yield env.timeout(delay)
            delay = np.random.uniform(low=15, high=30) / 60
            yield env.timeout(delay)

            # end of service at Theke 3 for this burger
            service_time = env.now - service_start
            stats.burgermeister_service_times.append(service_time)

            # total student waiting time (only for successfully completed burgers)
            if "order_time" in burger:
                stats.student_total_wait.append(env.now - burger["order_time"])

        print("sum", sum_deadline_violation)


# -------------------------------------------------------------------
# Burger creation with incremental ID
# -------------------------------------------------------------------
burger_id = 1


def create_burger():
    global burger_id
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
            burger['id'] = burger_id
            burger_id += 1
            return burger


# -------------------------------------------------------------------
# Run simulation
# -------------------------------------------------------------------
env = simpy.Environment()

bestellung_list = BestellungList(env, 0)
warm_schrank = WarmSchrank(env, 0)

env.process(Student(env, bestellung_list))
env.process(Zuarbeiter(env, bestellung_list, warm_schrank, name="Z1", stats=stats))
env.process(Zuarbeiter(env, bestellung_list, warm_schrank, name="Z2", stats=stats))
#env.process(Zuarbeiter(env, bestellung_list, warm_schrank, name="Z1", stats=stats))
#env.process(Zuarbeiter(env, bestellung_list, warm_schrank, name="Z2", stats=stats))
env.process(Burgermeister(env, bestellung_list, warm_schrank, stats=stats))


env.run(until=SIM_DURATION)


# -------------------------------------------------------------------
# Evaluation of statistics
# -------------------------------------------------------------------
def safe_mean(values):
    return float(np.mean(values)) if values else float('nan')


avg_wait_before_zuarbeiter = safe_mean(stats.wait_before_zuarbeiter)
avg_wait_before_burgermeister = safe_mean(stats.wait_before_burgermeister)
avg_student_total_wait = safe_mean(stats.student_total_wait)
avg_burgermeister_service = safe_mean(stats.burgermeister_service_times)

print("\n=== Auswertung ===")
print(f"1) Durchschnittliche Wartezeit vor Zuarbeiter: {avg_wait_before_zuarbeiter:.2f} Minuten")
print(f"   Durchschnittliche Wartezeit vor Burgermeister (WarmSchrank-Queue): "
      f"{avg_wait_before_burgermeister:.2f} Minuten")

if avg_wait_before_zuarbeiter > avg_wait_before_burgermeister:
    print("   -> Engpass eher im Bereich Bestellung/Zuarbeiter.")
elif avg_wait_before_burgermeister > avg_wait_before_zuarbeiter:
    print("   -> Engpass eher im Bereich WarmSchrank/Burgermeister.")
else:
    print("   -> Keine klare Engpassverschiebung erkennbar (beide ähnlich).")

print(f"\n2) Durchschnittliche Mittagszeit an Theke 3 (Servicezeit Burgermeister): "
      f"{avg_burgermeister_service:.2f} Minuten")

print("\n3) Durchschnittlicher Leerlauf:")
for name, idle in stats.zuarbeiter_idle.items():
    print(f"   {name}: {idle:.2f} Minuten (≈ {idle / SIM_DURATION:.2%} der Simulationszeit)")
print(f"   Burgermeister: {stats.burgermeister_idle:.2f} Minuten "
      f"(≈ {stats.burgermeister_idle / SIM_DURATION:.2%} der Simulationszeit)")

print(f"\n4) Durchschnittliche Wartezeit der Studierenden auf ihren Burger: "
      f"{avg_student_total_wait:.2f} Minuten")


# -------------------------------------------------------------------
# Gantt chart: each burger ID from order -> etikettieren
# -------------------------------------------------------------------
if burger_gantt_data:
    # sort by start time (optional, for nicer ordering)
    burger_gantt_data.sort(key=lambda x: x["start"])

    ids = [entry["id"] for entry in burger_gantt_data]
    # preserve order of first appearance
    unique_ids = list(dict.fromkeys(ids))
    id_to_row = {bid: i for i, bid in enumerate(unique_ids)}

    fig, ax = plt.subplots(figsize=(10, max(4, len(unique_ids) * 0.2)))

    for entry in burger_gantt_data:
        bid = entry["id"]
        row = id_to_row[bid]
        start = entry["start"]
        end = entry["end"]
        duration = end - start
        ax.barh(row, duration, left=start)

    ax.set_yticks(range(len(unique_ids)))
    ax.set_yticklabels([f"Burger {bid}" for bid in unique_ids])
    ax.set_xlabel("Time (minutes)")
    ax.set_title("Gantt chart: Order → Etikettieren per burger")

    plt.tight_layout()
    plt.show()
else:
    print("\nNo completed burgers to plot in Gantt chart.")
