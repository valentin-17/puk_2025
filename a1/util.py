import numpy as np
import simpy
from simpy import  PriorityStore
from typing import (
    Any,
    NamedTuple,
)

# TODO global variabel which alerts each process (Student, Zubereiter, Burgermeister)

np.random.seed(42)      # set the seed
SIM_DURATION = 100



class DynamicPriorityItem(NamedTuple):
    """Wrap an arbitrary *item* with an order-able *priority*.

    """

    burger: Any


    timestamp: Any

    env: Any

    def get_priority(self):
        return self.timestamp - self.env.now

    def __lt__(  # type: ignore[override]
        self, other
    ) -> bool:
        priorty_1 =self.timestamp - self.env.now
        priorty_2 =other.timestamp - other.env.now
        return priorty_1 < priorty_2

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
        self.store = simpy.FilterStore(env)

    def latency(self, value):
        yield self.env.timeout(self.delay)
        self.store.put(value)

    def put(self, value):
        self.env.process(self.latency(value))
    
    def get(self, filter_func=None):
        return self.store.get(filter=filter_func)


def Student(env, bestellung_list):
    
    while True:

        yield env.timeout(5)    # TODO mean timeout 
        burger_creation = DynamicPriorityItem(["warmeZutaten",2,3,4,5], timestamp=env.now+30, env=env)   # TODO set random burger generation from statistics
        #print("PRIO", burger_creation.get_priority())
        bestellung_list.put(burger_creation)


def Zuarbeiter(env, bestellung_list, warm_schrank):
    
    while True:

        bestellung = yield bestellung_list.get()
        yield env.timeout(5)
        print(f'Received this at {env.now} while {bestellung}')
        warm_schrank.put(bestellung)


def Burgermeister(env, bestellung_list, warm_schrank):
    
    while True:             
        
        # bekommt Bestellung
        yield env.timeout(1)
        warme_zutaten = yield warm_schrank.get(lambda x: x.get_priority() < 5)
        
        print("PRIO",warme_zutaten.get_priority())
        
        if np.random.rand() <= 0.05: #0.05
            p1 = DynamicPriorityItem("do homework", timestamp=warme_zutaten.get_priority(), env=env) 
            print("Fehler der Zuberetiung")
            
            bestellung_list.put(p1)
        else:
            print(f'Deliverd at time: {env.now}')
            print("liefern")
        



# Setup and start the simulation
print('Event Latency')
env = simpy.Environment()

bestellung_list = BestellungList(env, 10)
warm_schrank = WarmSchrank(env, 5)
env.process(Student(env, bestellung_list))
env.process(Zuarbeiter(env, bestellung_list, warm_schrank))
env.process(Burgermeister(env, bestellung_list, warm_schrank))

env.run(until=SIM_DURATION)