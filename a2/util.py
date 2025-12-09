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
SIM_DURATION = 200 # von 11:00 bis 14:30
TRESHOLD_REFILL = 0.001


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
        #print("bestellt at ", env.now)
        delay = np.random.normal(loc=105, scale=8) / 60
        yield env.timeout(delay)


def Zuarbeiter(env, bestellung_list, warm_schrank, station_tank, bacon):
    
    while True:

        bestellung = yield bestellung_list.get()
        delay = np.random.gamma(shape=10, scale=2) / 60
        yield env.timeout(delay)
        delay = np.random.uniform(low=360, high=600) / 60
        yield env.timeout(delay)
        

        # Get the required amount of fuel
        fuel_required = bestellung.get_burger()['patty']
        #print(bestellung.get_burger()['patty'])
        print(fuel_required)
        print("patty level", station_tank.level)
        if fuel_required > 0:
            yield station_tank.get(fuel_required)
            if station_tank.level / station_tank.capacity < TRESHOLD_REFILL:
                # We need to call the tank truck now!
                print(f'{env.now:6.1f} s: Calling truck')
                # Wait for the tank truck to arrive and refuel the station tank
                if station_tank.level > 0:
                    env.process(tank_truck(env, station_tank))
                else:
                    print("wait")
                    yield env.process(tank_truck(env, station_tank))
                    print("stopped waiting")
            # The "actual" refueling process takes some time
            yield env.timeout(1)

        
                # Get the required amount of fuel
        fuel_required = bestellung.get_burger()['bacon']
        #print(bestellung.get_burger()['patty'])
        print(fuel_required)
        print("bacon level", bacon.level)
        if fuel_required > 0:
            yield bacon.get(fuel_required)
            if bacon.level / bacon.capacity < TRESHOLD_REFILL:
                # We need to call the tank truck now!
                print(f'{env.now:6.1f} s: Calling truck')
                # Wait for the tank truck to arrive and refuel the station tank
                if bacon.level > 0:
                    env.process(tank_truck(env, bacon))
                else:
                    yield env.process(tank_truck(env, bacon))
            # The "actual" refueling process takes some time
            yield env.timeout(1)

        #print(f'{env.now:6.1f} s: refueled with {fuel_required:.1f}L')


        warm_schrank.put(bestellung)

def Burgermeister(env, bestellung_list, warm_schrank):
    sum = 0
    while True:             
        
        delay = np.random.normal(loc=5, scale=1) / 60
        yield env.timeout(delay)

        warme_zutaten = yield warm_schrank.get(lambda x: x.get_deadline() < 5)
        #print("deadline lower 5: ", warme_zutaten.get_deadline())
        #print(warme_zutaten.get_burger())

        if np.random.rand() <= 0.5:
            #print("Fehler der Zuberetiung")
            bestellung_list.put(warme_zutaten)
        else:

            #print("etikitieren zur deadline: ", warme_zutaten.get_deadline())
            if warme_zutaten.get_deadline() < 0:
                sum += warme_zutaten.get_deadline()
            delay = np.random.uniform(low=10, high=20) / 60
            yield env.timeout(delay)
            # if pommes
            delay = np.random.uniform(low=15, high=30) / 60
            yield env.timeout(delay)
        #print("sum", sum)

        

def tank_truck(env, station_tank):
    """Arrives at the gas station after a certain delay and refuels it."""
    yield env.timeout(0.1)
    print("tank truck level ", station_tank.level)
    amount = station_tank.capacity - station_tank.level
    station_tank.put(amount)
    print(
        f'{env.now:6.1f} s: Tank truck arrived and refuelled station with {amount:.1f}L'
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



env = simpy.Environment()

bestellung_list = BestellungList(env, 5)
warm_schrank = WarmSchrank(env, 10)

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

gas_station = simpy.Resource(env, 1)
#env.process(gas_station_control(env, salad))
env.process(Zuarbeiter(env, bestellung_list, warm_schrank, patty, bacon))
env.process(Zuarbeiter(env, bestellung_list, warm_schrank, patty, bacon))
#env.process(Zuarbeiter(env, bestellung_list, warm_schrank))
env.process(Burgermeister(env, bestellung_list, warm_schrank))

env.run(until=SIM_DURATION)