import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, expon, uniform

from a3.config import PROCESSING_TIME

C_MAP = {
    'a': '#840853',
    'b': '#ba55aa',
    'c': '#d1a7ef',
    'd': '#3a609c',
    'e': '#69a0cb',
}

x = np.linspace(0, 160, 2000)

for model, params in PROCESSING_TIME.items():
    plt.figure(figsize=(8, 4))

    for name, spec in params.items():
        if spec[0] == 'normal':
            mu, sigma = spec[1], spec[2]
            y = norm.pdf(x, mu, sigma)

        elif spec[0] == 'exponential':
            scale = spec[1]
            y = expon.pdf(x, scale=scale)

        elif spec[0] == 'uniform':
            low, high = spec[1], spec[2]
            y = uniform.pdf(x, low, high - low)

        plt.plot(x, y, label=f"{name} ({spec[0]})", color=C_MAP[name])

    plt.title(f"Verteilungen für {model}")
    plt.xlabel("Zeit in s")
    plt.ylim(0, 0.21)
    plt.legend(title='Variante')
    plt.tight_layout()
    plt.savefig(f"figures/{model}_distributions.png")