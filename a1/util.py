import numpy as np
import simpy
from simpy import  PriorityStore, FilterStore, Store
from simpy.resources.store import FilterStoreGet, StorePut
from typing import (
    Any,
    NamedTuple,
    Optional,
    Callable
)
from heapq import heappush
import config as c

np.random.seed(42)      # set the seed
SIM_DURATION = 100 # von 11:00 bis 14:30



class DynamicDeadlineItem(NamedTuple):
    """Wrap an arbitrary *item* with an order-able *deadline*.
    """
    burger: Any

    deadline: Any

    env: Any

    def get_burger(self):
        return self.burger

    def get_deadline(self):
        return self.deadline - self.env.now

    def __lt__(  # type: ignore[override]
        self, other
    ) -> bool:
        priorty_1 =self.deadline - self.env.now
        priorty_2 =other.deadline - other.env.now
        return priorty_1 < priorty_2


class PriorityFilterStore(FilterStore):

    def _do_put(self, event: StorePut) -> Optional[bool]:
        if len(self.items) < self._capacity:
            heappush(self.items, event.item)
            event.succeed()
        return None
    
    

class BestellungList:
    
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
    
    def __init__(self, env, delay):
        self.env = env
        self.delay = delay
        self.store = PriorityFilterStore(env) # sum = -1864 for p_of_failure = 0.5
        #self.store = FilterStore(env) # sum = -1871 for p_of_failure = 0,5 

    def latency(self, value):
        yield self.env.timeout(self.delay)
        self.store.put(value)

    def put(self, value):
        self.env.process(self.latency(value))
    
    def get(self, filter_func=None):
        return self.store.get(filter=filter_func)


def Student(env, bestellung_list):
    
    while env.now < 180:
        
        burger_creation = DynamicDeadlineItem(create_burger(), deadline=env.now+30, env=env)
        bestellung_list.put(burger_creation)
        print("bestellt at ", env.now)
        delay = np.random.normal(loc=105, scale=8) / 60
        yield env.timeout(delay)


def Zuarbeiter(env, bestellung_list, warm_schrank):
    
    while True:

        bestellung = yield bestellung_list.get()
        delay = np.random.gamma(shape=10, scale=2) / 60
        yield env.timeout(delay)
        delay = np.random.uniform(low=360, high=600) / 60
        yield env.timeout(delay)
        warm_schrank.put(bestellung)


def Burgermeister(env, bestellung_list, warm_schrank):
    sum = 0
    while True:             
        
        delay = np.random.normal(loc=5, scale=1) / 60
        yield env.timeout(delay)

        warme_zutaten = yield warm_schrank.get(lambda x: x.get_deadline() < 5)
        print("deadline lower 5: ", warme_zutaten.get_deadline())
        print(warme_zutaten.get_burger())

        if np.random.rand() <= 0.5:
            print("Fehler der Zuberetiung")
            bestellung_list.put(warme_zutaten)
        else:

            print("etikitieren zur deadline: ", warme_zutaten.get_deadline())
            if warme_zutaten.get_deadline() < 0:
                sum += warme_zutaten.get_deadline()
            delay = np.random.uniform(low=10, high=20) / 60
            yield env.timeout(delay)
            # if pommes
            delay = np.random.uniform(low=15, high=30) / 60
            yield env.timeout(delay)
        print("sum", sum)

        

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



env = simpy.Environment()

bestellung_list = BestellungList(env, 5)
warm_schrank = WarmSchrank(env, 10)
env.process(Student(env, bestellung_list))
env.process(Zuarbeiter(env, bestellung_list, warm_schrank))
env.process(Zuarbeiter(env, bestellung_list, warm_schrank))
env.process(Burgermeister(env, bestellung_list, warm_schrank))

env.run(until=SIM_DURATION)