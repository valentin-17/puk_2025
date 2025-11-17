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

np.random.seed(42)      # set the seed
SIM_DURATION = 210 # von 10:00 bis 14:30



class DynamicDeadlineItem(NamedTuple):
    """Wrap an arbitrary *item* with an order-able *deadline*.

    """
    burger: Any

    deadline: Any

    env: Any

    def get_deadline(self):
        return self.deadline - self.env.now

    def __lt__(  # type: ignore[override]
        self, other
    ) -> bool:
        priorty_1 =self.deadline - self.env.now
        priorty_2 =other.deadline - other.env.now
        return priorty_1 < priorty_2


class PriorityFilterStore(Store):

    def _do_put(self, event: StorePut) -> Optional[bool]:
        if len(self.items) < self._capacity:
            heappush(self.items, event.item)
            event.succeed()
        return None
    
    def get(
        self, filter: Callable[[Any], bool] = lambda item: True
    ) -> FilterStoreGet:
        """Request to get an *item*, for which *filter* returns ``True``,
        out of the store."""
        return FilterStoreGet(self, filter)
    

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
        #self.store = PriorityFilterStore(env) # sum = -1864 for p_of_failure = 0.5
        self.store = FilterStore(env) # sum = -1871 for p_of_failure = 0,5 

    def latency(self, value):
        yield self.env.timeout(self.delay)
        self.store.put(value)

    def put(self, value):
        self.env.process(self.latency(value))
    
    def get(self, filter_func=None):
        return self.store.get(filter=filter_func)


def Student(env, bestellung_list):
    
    while env.now < 180:

        burger_creation = DynamicDeadlineItem(["warmeZutaten",2,3,4,5], deadline=env.now+30, env=env)   # TODO set random burger generation from statistics
        # Burgers have *at least 2 but at max 20 ingredients*
        bestellung_list.put(burger_creation)
        #print("bestellt at ", env.now)
        #print("deadline",burger_creation.get_deadline())
        yield env.timeout(2)    # Time between orders in seconds: *Normal(mean=105, scale=8)* 


def Zuarbeiter(env, bestellung_list, warm_schrank):
    
    while True:

        bestellung = yield bestellung_list.get()

        yield env.timeout(5) # TODO Warm ingredients have a preparation time of *Uniform(360, 600) seconds (i.e. 6 - 10 min.)* independent of total 
        # TODO something like this: yield env.timeout(get_preperation_time(bestellung['warmeZutaten']))
        # TODO Worker needs *Gamma(shape=10, scale=2) seconds (i.e. 10 - 30 sec.)* to take ingredients from freezer
        # TODO Burger is assembled *at max 5 minutes before pick up time*
        warm_schrank.put(bestellung)


def Burgermeister(env, bestellung_list, warm_schrank):
    sum = 0
    while True:             
        
        # TODO Cold ingredients take *Normal(mean=5, scale=1) seconds (i.e. approx. 5 sec.)*
        yield env.timeout(0.083)
        warme_zutaten = yield warm_schrank.get(lambda x: x.get_deadline() < 5)
        
        print("PRIO",warme_zutaten.get_deadline())

        if np.random.rand() <= 0.5:
        #if np.random.rand() <= 0.5:
            print("Fehler der Zuberetiung")
            bestellung_list.put(warme_zutaten)
        else:
            #print(f'Deliverd at time: {env.now}')
            print("etikitieren")
            # TODO etikitieren
            if warme_zutaten.get_deadline() < 0:
                sum += warme_zutaten.get_deadline()

        print("sum", sum)

        
        



# Setup and start the simulation
#print('Event Latency')
env = simpy.Environment()

bestellung_list = BestellungList(env, 10)
warm_schrank = WarmSchrank(env, 5)
env.process(Student(env, bestellung_list))
env.process(Zuarbeiter(env, bestellung_list, warm_schrank))
env.process(Zuarbeiter(env, bestellung_list, warm_schrank))
env.process(Burgermeister(env, bestellung_list, warm_schrank))

env.run(until=SIM_DURATION)