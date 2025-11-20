# Framework
## Task 1
- Timespan *11:30 - 14:30*
- Burger can be ordered between *11:00 and 14:00*
- Pick up is exactly *30 minutes after order*
- *30 - 40 orders/h*
- Time between orders in seconds: *Normal(mean=105, scale=8)*
- Burgers have *at least 2 but at max 20 ingredients*
- Warm ingredients have a preparation time of *Uniform(360, 600) seconds (i.e. 6 - 10 min.)* independent of total ingredients
- Worker needs *Gamma(shape=10, scale=2) seconds (i.e. 10 - 30 sec.)* to take ingredients from freezer
- Burger is assembled *at max 5 minutes before pick up time*
- Toasting buns takes *40 seconds*
- Cold ingredients take *Normal(mean=5, scale=1) seconds (i.e. approx. 5 sec.)*
- *5%* of burgers fail during assembly, assembly has to restart
- Packing takes *Uniform(10, 20) seconds*
- *50%* of students order fries
- If fries are ordered packing takes *Uniform(15, 30)*
- Fries are always ready to pack and do not have to be prepared
- Received orders are fulfilled in the same Timespan (i.e. on the same day)
## Task 2
- There is only 1 helper, The helper is not a linecook
- The helper can only carry 1 ingredient per trip
- A trip takes *Normal(mean=3x60, scale=0.5x60)* seconds
- The finite bins at prep and assembly station have 30 slots
- The s,Q policy has a safety threshold of 15%
- The s,Q policy has a notification threshold of 30%
- The order quantity is the delta missing in a bin (i.e. bin.max - bin.current)

# How the task has been tackled
## Finding distributions
To find the distributions first the integrated data viewer in Pycharm has been used to get a visual impression of the data distribution. Then the columns have been pre classified manually based on their shape. The result is as follows: The columns ['Bun', 'Patty', 'Bacon', 'Salat', 'Sauce'] follow a binomial distribution. The columns ['Gewuerz', 'Kaese', 'Gemuese'] follow a poisson distribution. This information was then used to construct a scipy fit method which determined the parameters of the distributions for each ingredient. The resulting parameters are listed below:

| Ingredient | Distribution | params        |
|:-----------|:-------------|:--------------|
| Bun        | Binomial     | [n=1, p=1]    |
| Patty      | Binomial     | [n=2, p=0.74] |
| Gewuerz    | Poisson      | [mu=0.98]     |
| Kaese      | Poisson      | [mu=0.90]     |
| Bacon      | Binomial     | [n=3, p=0.30] |
| Salat      | Binomial     | [n=3, p=0.33] |
| Gemuese    | Poisson      | [mu=1.97]     |
| Sauce      | Binomial     | [n=3, p=0.66] |

## Generating artificial data
To generate more samples a simple utility function has been written to generate an arbitrary amount of artificial burger orders which can be used in simulation.