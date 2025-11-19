"""

GEMINI SOLUTION

DO NOT TURN IN

"""

import simpy
import numpy as np
import random
from collections import namedtuple

# --- Simulation Parameters ---

# Staffing
NUM_ASSISTANTS = 2
NUM_COOKS = 1

# Order timing (all times in seconds)
ORDER_WINDOW = 3 * 3600       # 3 hours (11:00 AM to 2:00 PM)
PICKUP_LEAD_TIME = 30 * 60    # 30 minutes
ASSEMBLY_START_OFFSET = 5 * 60 # 5 minutes before pickup

# Inter-arrival time for orders
MEAN_INTERARRIVAL_TIME = 105
STD_DEV_INTERARRIVAL_TIME = 8

# Burger characteristics
MIN_INGREDIENTS = 2
MAX_INGREDIENTS = 20
FRIES_PROBABILITY = 0.5
FAILURE_RATE = 0.05

# Process times (seconds)
FREEZER_TIME_SHAPE = 10
FREEZER_TIME_SCALE = 2
HOT_PREP_TIME_MIN = 360
HOT_PREP_TIME_MAX = 600
TOASTING_TIME = 40
ASSEMBLY_TIME_PER_INGREDIENT_MEAN = 5
ASSEMBLY_TIME_PER_INGREDIENT_STD = 1
PACKAGING_TIME_MIN = 10
PACKAGING_TIME_MAX = 20
FRIES_PACKAGING_TIME_MIN = 15
FRIES_PACKAGING_TIME_MAX = 30

# Simulation duration
SIMULATION_TIME = 5 * 3600  # Run for 5 hours to ensure all orders are processed

# Data collection
OrderData = namedtuple('OrderData', ['order_time', 'pickup_time', 'completion_time', 'rework_count'])
ResourceUsage = namedtuple('ResourceUsage', ['resource_name', 'start_time', 'end_time'])

class BurgerRestaurant:
    def __init__(self, env):
        self.env = env
        self.assistants = simpy.Resource(env, capacity=NUM_ASSISTANTS)
        self.cooks = simpy.Resource(env, capacity=NUM_COOKS)
        self.order_stats = []
        self.assistant_usage = []
        self.cook_usage = []

def burger_process(env, order_id, restaurant):
    """Models the entire lifecycle of a single burger order."""
    order_time = env.now
    pickup_time = order_time + PICKUP_LEAD_TIME
    rework_count = 0

    while True: # Loop to handle the 5% rework chance
        # --- 1. Prep Counter (Assistants) ---
        prep_start_time = env.now
        with restaurant.assistants.request() as req:
            yield req
            # Record assistant usage
            assistant_start_use = env.now
            
            # Time to get ingredients from freezer
            yield env.timeout(np.random.gamma(FREEZER_TIME_SHAPE, FREEZER_TIME_SCALE))
            
            # Time to prepare hot ingredients
            yield env.timeout(random.uniform(HOT_PREP_TIME_MIN, HOT_PREP_TIME_MAX))
            
            restaurant.assistant_usage.append(ResourceUsage('assistant', assistant_start_use, env.now))

        # --- 2. Wait for Assembly Window ---
        assembly_earliest_start = pickup_time - ASSEMBLY_START_OFFSET
        if env.now < assembly_earliest_start:
            yield env.timeout(assembly_earliest_start - env.now)

        # --- 3. Preparation Counter (Cook) ---
        with restaurant.cooks.request() as req:
            yield req
            cook_start_use = env.now

            # Toasting buns
            yield env.timeout(TOASTING_TIME)

            # Assembling burger
            num_ingredients = random.randint(MIN_INGREDIENTS, MAX_INGREDIENTS)
            # Assuming 2 hot ingredients (patty, bun), rest are cold
            num_cold_ingredients = num_ingredients - 2 
            if num_cold_ingredients > 0:
                assembly_time = sum(np.random.normal(ASSEMBLY_TIME_PER_INGREDIENT_MEAN, ASSEMBLY_TIME_PER_INGREDIENT_STD) for _ in range(num_cold_ingredients))
                yield env.timeout(max(0, assembly_time))

            # --- 4. Quality Control ---
            if random.random() < FAILURE_RATE:
                rework_count += 1
                restaurant.cook_usage.append(ResourceUsage('cook', cook_start_use, env.now))
                # Continue to next iteration of the while loop to restart the process
                continue

            # --- 5. Packaging ---
            packaging_time = random.uniform(PACKAGING_TIME_MIN, PACKAGING_TIME_MAX)
            if random.random() < FRIES_PROBABILITY:
                packaging_time += random.uniform(FRIES_PACKAGING_TIME_MIN, FRIES_PACKAGING_TIME_MAX)
            yield env.timeout(packaging_time)
            
            restaurant.cook_usage.append(ResourceUsage('cook', cook_start_use, env.now))
            
            # If successful, break the rework loop
            break
    
    completion_time = env.now
    restaurant.order_stats.append(OrderData(order_time, pickup_time, completion_time, rework_count))

def order_generator(env, restaurant):
    """Generates burger orders during the ordering window."""
    order_id = 0
    while env.now < ORDER_WINDOW:
        env.process(burger_process(env, order_id, restaurant))
        order_id += 1
        
        interarrival_time = np.random.normal(MEAN_INTERARRIVAL_TIME, STD_DEV_INTERARRIVAL_TIME)
        yield env.timeout(max(0, interarrival_time))

def run_simulation():
    """Sets up and runs the simulation, then prints results."""
    env = simpy.Environment()
    restaurant = BurgerRestaurant(env)
    env.process(order_generator(env, restaurant))
    env.run(until=SIMULATION_TIME)

    # --- Analysis ---
    print("\n--- Burger Restaurant Simulation Results ---")

    total_orders = len(restaurant.order_stats)
    if total_orders == 0:
        print("No orders were completed during the simulation.")
        return

    # 1. Bottlenecks (by analyzing wait times, which simpy does implicitly via utilization)
    # We calculate utilization to infer bottlenecks.
    total_sim_time = restaurant.env.now
    
    assistant_busy_time = sum(usage.end_time - usage.start_time for usage in restaurant.assistant_usage)
    assistant_utilization = (assistant_busy_time / (NUM_ASSISTANTS * total_sim_time)) * 100
    
    cook_busy_time = sum(usage.end_time - usage.start_time for usage in restaurant.cook_usage)
    cook_utilization = (cook_busy_time / (NUM_COOKS * total_sim_time)) * 100

    print("\n1. Bottleneck Analysis (Resource Utilization):")
    print(f"  - Assistants Utilization: {assistant_utilization:.2f}%")
    print(f"  - Burger Cook Utilization: {cook_utilization:.2f}%")
    if assistant_utilization > 90 or cook_utilization > 90:
        bottleneck = "Assistants" if assistant_utilization > cook_utilization else "Burger Cook"
        print(f"  -> Conclusion: The {bottleneck} area is a significant bottleneck (>90% utilization).")
    else:
        print("  -> Conclusion: No critical bottlenecks detected (all resources <90% utilization).")


    # 2. Average Lunchtime
    # This is defined by the simulation parameters. The simulation runs until all orders are processed.
    print(f"\n2. Lunchtime Information:")
    print(f"  - Order period: 11:00 AM to 2:00 PM (3 hours).")
    print(f"  - Canteen open for pickup: 11:30 AM to 2:30 PM (and beyond until orders are filled).")
    print(f"  - Total time to process all orders: {total_sim_time / 3600:.2f} hours from the start of the shift.")

    # 3. Average Free Time
    assistant_free_time_percent = 100 - assistant_utilization
    cook_free_time_percent = 100 - cook_utilization
    print("\n3. Average Staff Free Time:")
    print(f"  - Assistants were free {assistant_free_time_percent:.2f}% of the time.")
    print(f"  - Burger Cook was free {cook_free_time_percent:.2f}% of the time.")

    # 4. Average Guest Wait Time
    # Wait time is the time from the scheduled pickup to actual completion.
    guest_wait_times = [max(0, data.completion_time - data.pickup_time) for data in restaurant.order_stats]
    avg_wait_time_seconds = np.mean(guest_wait_times)
    
    print("\n4. Average Guest Wait Time:")
    print(f"  - A guest waits on average {avg_wait_time_seconds / 60:.2f} minutes for their burger after the scheduled pickup time.")
    print("------------------------------------------\n")


if __name__ == '__main__':
    run_simulation()
