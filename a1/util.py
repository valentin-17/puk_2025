import numpy as np
import simpy
from simpy import  PriorityStore, FilterStore, Store
from simpy.resources.store import FilterStoreGet, StorePut
from typing import (
    Any,
    NamedTuple,
    Optional,
    Callable,
    Union,
    Sequence
)
from heapq import heappush
import config as c
from simpy.resources.container import Container

from simpy.core import BoundClass, Environment
from simpy.resources import base

ContainerAmount = Union[int, float]

np.random.seed(42)      # set the seed
SIM_DURATION = 800 # von 11:00 bis 14:30
TRESHOLD_REFILL = 0.1


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


class Vorratsbehaelter(Container):
    
    def __init__(
        self,
        env: Environment,
        capacity: ContainerAmount = float('inf'),
        init: ContainerAmount = 0,
        zutaten_list : Optional[Sequence[str]] = None
    ):
        super().__init__(env, capacity=capacity, init=init)
        self.zutaten_list: list[str] = zutaten_list

    def get_zutaten_list(self) -> Optional[Sequence[str]]:
        return self.zutaten_list


def Student(env, bestellung_list):
    
    while env.now < 180:
        
        burger_creation = DynamicDeadlineItem(create_burger(), deadline=env.now+30, env=env)
        bestellung_list.put(burger_creation)
        print("bestellt at ", env.now)
        delay = np.random.normal(loc=105, scale=8) / 60
        yield env.timeout(delay)
        print("id", burger_creation.get_burger()['id'])



def Zuarbeiter(env, bestellung_list, warm_schrank, truck_resource, patty, bacon):
    
    while True:

        bestellung = yield bestellung_list.get()
        delay = np.random.gamma(shape=10, scale=2) / 60
        yield env.timeout(delay)
        delay = np.random.uniform(low=360, high=600) / 60
        yield env.timeout(delay)
        

        # patty
        yield from handle_ingredient(env, bestellung, "patty", patty, truck_resource)

        # bacon
        yield from handle_ingredient(env, bestellung, "bacon", bacon, truck_resource)


        warm_schrank.put(bestellung)

def Burgermeister(env, bestellung_list, warm_schrank, truck_resource, bun, salad, cheese, vegetables):
    sum_deadline = 0
    AGE_THRESHOLD = 5   # your condition: start_order - now < -25
    POLL_STEP = 1         # minutes between checks (can be smaller if you like)

    while True:

        delay = np.random.normal(loc=5, scale=1) / 60
        yield env.timeout(delay)

        while True:
            get_event = warm_schrank.get(
                lambda x: x.get_deadline() < AGE_THRESHOLD
            )
            timeout_event = env.timeout(POLL_STEP)

            result = yield get_event | timeout_event

            if get_event in result:
                warme_zutaten = result[get_event]
                break
            else:
                get_event.cancel()
                

        # from here on we have a matching item
        #print("deadline lower -25: ", warme_zutaten.get_start_order(), "-", env.now)
        #print("deadline lower -25: ", warme_zutaten.get_start_order() - env.now)

        if np.random.rand() <= 0.05:
            # preparation failed -> reinsert into order list
            bestellung_list.put(warme_zutaten)
        else:
            if warme_zutaten.get_deadline() < 0:
                sum_deadline += warme_zutaten.get_deadline()

            # bun
            yield from handle_ingredient(env, warme_zutaten, "bun", bun, truck_resource)

            # salad
            yield from handle_ingredient(env, warme_zutaten, "salad", salad, truck_resource)

            # cheese
            yield from handle_ingredient(env, warme_zutaten, "cheese", cheese, truck_resource)

            # vegetables
            yield from handle_ingredient(env, warme_zutaten, "vegetables", vegetables, truck_resource)

            delay = np.random.uniform(low=10, high=20) / 60
            yield env.timeout(delay)
            delay = np.random.uniform(low=15, high=30) / 60
            yield env.timeout(delay)
            print(env.now, ": ", warme_zutaten.get_burger()['id'])

        # optional:
        # print("sum_deadline", sum_deadline)


        

def tank_truck(env, container, truck_resource):
    """Single global tank truck that refills the given container."""
    # Only one truck at a time:
    with truck_resource.request() as req:
        # wait until truck is free
        yield req

        # drive time / arrival delay
        yield env.timeout(1)

        print("tank truck level before:", container.level)
        amount = container.capacity - container.level
        if amount > 0:
            yield container.put(amount)
        print(
            f'{env.now:6.1f} s: Tank truck refuelled container with {amount}'
        )


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


def handle_ingredient(env, bestellung, key, container, truck_resource):
    """Take needed amount of `key` from `container` and refill if needed."""
    fuel_required = bestellung.get_burger()[key]
    print(f"{key} needed:", fuel_required)
    print(f"{key} level:", container.level)

    if fuel_required <= 0:
        return  # nothing to do

    # If low: start a truck in the background
    if container.level / container.capacity < TRESHOLD_REFILL:
        print(f'{env.now:6.1f} s: Calling truck for {key}')
        env.process(tank_truck(env, container, truck_resource))

    # Take from container (may block until truck has refilled)
    yield container.get(fuel_required)


env = simpy.Environment()

bestellung_list = BestellungList(env, 0)
warm_schrank = WarmSchrank(env, 0)

env.process(Student(env, bestellung_list))
bun = Vorratsbehaelter(env, capacity=30, init=30, zutaten_list= ['bun','bun sesam'])
patty = Vorratsbehaelter(env, capacity=30, init=30, zutaten_list= ['patty'])
bacon = Vorratsbehaelter(env, capacity=30, init=30, zutaten_list= ['bacon'])
salad = Vorratsbehaelter(env, capacity=30, init=30, zutaten_list= ['salad'])
#sauce = Vorratsbehaelter(env, capacity=30, init=30, zutaten_list= ['S'])
#spice = Vorratsbehaelter(env, capacity=30, init=30, zutaten_list= ['S'])
cheese = Vorratsbehaelter(env, capacity=30, init=30, zutaten_list= ['cheese'])
vegetables = Vorratsbehaelter(env, capacity=30, init=30, zutaten_list= ['vegetables'])
print(salad.get_zutaten_list())



truck_resource = simpy.Resource(env, capacity=1)

env.process(Zuarbeiter(env, bestellung_list, warm_schrank, truck_resource, patty, bacon))
env.process(Zuarbeiter(env, bestellung_list, warm_schrank, truck_resource, patty, bacon))
env.process(Zuarbeiter(env, bestellung_list, warm_schrank, truck_resource, patty, bacon))
env.process(Zuarbeiter(env, bestellung_list, warm_schrank, truck_resource, patty, bacon))
env.process(Zuarbeiter(env, bestellung_list, warm_schrank, truck_resource, patty, bacon))
env.process(Zuarbeiter(env, bestellung_list, warm_schrank, truck_resource, patty, bacon))

env.process(Burgermeister(env, bestellung_list, warm_schrank, truck_resource, bun, salad, cheese, vegetables))

env.run(until=SIM_DURATION)