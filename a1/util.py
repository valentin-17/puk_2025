"""
Event Latency by Keith Smith

"""

import simpy


# TODO global variabel which alerts each process (Student, Zubereiter, Burgermeister)


SIM_DURATION = 100


class BestellungList:
    

    def __init__(self, env, delay):
        self.env = env
        self.delay = delay
        self.store = simpy.Store(env)

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
        self.store = simpy.Store(env)

    def latency(self, value):
        yield self.env.timeout(self.delay)
        self.store.put(value)

    def put(self, value):
        self.env.process(self.latency(value))

    def get(self):
        return self.store.get()


def Student(env, bestellung_list):
    
    while True:

        yield env.timeout(5)    # TODO mean timeout 
        burger_creation = ["warmeZutaten",2,3,4,5]   # TODO set random burger generation from statistics
        bestellung_list.put(burger_creation)


def Zuarbeiter(env, bestellung_list, warm_schrank):
    
    while True:

        bestellung = yield bestellung_list.get()
        yield env.timeout(5)
        print(f'Received this at {env.now} while {bestellung}')
        warm_schrank.put(bestellung[0])


def Burgermeister(env, bestellung_list, warm_schrank):
    
    while True:             
        
        # bekommt Bestellung

        warme_zutaten = yield warm_schrank.get()
        
        print(warme_zutaten)

        yield env.timeout(5)

        """
        if random <= 0.05:
            print("Fehler der Zuberetiung")
            bestellung_list.put("die alte Bestllung") # TODO set high Priority
        """

        print(f'Received this at {env.now}')
        print("liefern")
        



# Setup and start the simulation
print('Event Latency')
env = simpy.Environment()

bestellung_list = BestellungList(env, 10)
warm_schrank = WarmSchrank(env, 10)
env.process(Student(env, bestellung_list))
env.process(Zuarbeiter(env, bestellung_list, warm_schrank))
env.process(Burgermeister(env, bestellung_list, warm_schrank))

env.run(until=SIM_DURATION)