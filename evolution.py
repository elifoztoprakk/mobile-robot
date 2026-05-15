"""
evolution.py
============
(mu + lambda) Evolution Strategy for evolving robot controllers.

A (mu + lambda)-ES works as follows each generation:
  1. From the mu parents, sample lambda offspring by adding
     Gaussian noise (sigma * N(0,I)) to a randomly chosen parent.
  2. Evaluate all mu + lambda individuals.
  3. Keep the mu best as parents for the next generation.

Sigma (the mutation step size) is adapted using the 1/5-success rule:
if more than 1/5 of offspring were better than the worst parent,
increase sigma; otherwise decrease it.  This is the simplest form of
self-adaptation and is easy to explain in a report.

Also contains:
  - Rastrigin and Rosenbrock benchmark functions
  - run_benchmark()  : optimise a benchmark and return a convergence curve
  - plot_benchmarks(): plot both curves and save to PNG

Author: Person 4
"""

import math
import time
from typing import Callable, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark functions
# ═══════════════════════════════════════════════════════════════════════════════

def rastrigin(x: np.ndarray) -> float:
    """
    Rastrigin function — highly multimodal benchmark.
    Global minimum: f(0,...,0) = 0.
    Typical search range: x_i in [-5.12, 5.12].
    """
    n = len(x)
    A = 10.0
    return float(A * n + np.sum(x**2 - A * np.cos(2 * math.pi * x)))


def rosenbrock(x: np.ndarray) -> float:
    """
    Rosenbrock (banana) function — unimodal but narrow curved valley.
    Global minimum: f(1,...,1) = 0.
    Typical search range: x_i in [-2.048, 2.048].
    """
    return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2))


# ═══════════════════════════════════════════════════════════════════════════════
# (mu + lambda) Evolution Strategy
# ═══════════════════════════════════════════════════════════════════════════════

class EvolutionStrategy:
    """
    (mu + lambda) ES with 1/5-success-rule step-size adaptation.

    Parameters
    ----------
    fitness_fn   : callable(weights) -> float.  HIGHER is better.
                   The EA maximises this value.
    n_weights    : dimensionality of the search space.
    mu           : number of parents kept each generation.
    lam          : number of offspring generated each generation.
    sigma_init   : initial mutation standard deviation.
    sigma_min    : lower clamp on sigma (prevents premature convergence).
    sigma_max    : upper clamp on sigma.
    seed         : random seed for reproducibility.
    """

    # 1/5-success-rule constants
    _C_INC = 1.22   # multiply sigma when success rate > 1/5
    _C_DEC = 0.82   # multiply sigma when success rate < 1/5

    def __init__(
        self,
        fitness_fn:  Callable[[np.ndarray], float],
        n_weights:   int,
        mu:          int   = 10,
        lam:         int   = 50,
        sigma_init:  float = 0.5,
        sigma_min:   float = 1e-4,
        sigma_max:   float = 5.0,
        seed:        int   = 42,
    ):
        self.fitness_fn  = fitness_fn
        self.n           = n_weights
        self.mu          = mu
        self.lam         = lam
        self.sigma       = sigma_init
        self.sigma_min   = sigma_min
        self.sigma_max   = sigma_max
        self.rng         = np.random.default_rng(seed)

        # Initialise population: mu individuals drawn from N(0, sigma_init)
        self.population: List[np.ndarray] = [
            self.rng.normal(0.0, sigma_init, n_weights)
            for _ in range(mu)
        ]
        self.fitnesses: List[float] = [fitness_fn(ind) for ind in self.population]

        # Logging
        self.best_fitness_history: List[float] = [max(self.fitnesses)]
        self.mean_fitness_history: List[float] = [float(np.mean(self.fitnesses))]
        self.sigma_history:        List[float] = [sigma_init]
        self.generation = 0

    # ------------------------------------------------------------------ #

    def step(self) -> Tuple[np.ndarray, float]:
        """
        Run one generation.

        Returns
        -------
        best_individual : weight vector of the current best individual.
        best_fitness    : its fitness value.
        """
        # 1. Generate lambda offspring
        offspring      = []
        offspring_fits = []
        successes      = 0

        worst_parent_fit = min(self.fitnesses)

        for _ in range(self.lam):
            # Pick a random parent
            idx    = self.rng.integers(0, self.mu)
            parent = self.population[idx]

            # Mutate
            child = parent + self.sigma * self.rng.standard_normal(self.n)
            fit   = self.fitness_fn(child)

            offspring.append(child)
            offspring_fits.append(fit)

            if fit > worst_parent_fit:
                successes += 1

        # 2. (mu + lambda) selection: keep best mu from parents + offspring
        combined      = self.population + offspring
        combined_fits = self.fitnesses  + offspring_fits

        ranked = sorted(zip(combined_fits, range(len(combined))), reverse=True)
        top_mu = ranked[:self.mu]

        self.population = [combined[i] for _, i in top_mu]
        self.fitnesses  = [combined_fits[i] for _, i in top_mu]

        # 3. 1/5-success-rule sigma adaptation
        success_rate = successes / self.lam
        if success_rate > 0.2:
            self.sigma = min(self.sigma * self._C_INC, self.sigma_max)
        else:
            self.sigma = max(self.sigma * self._C_DEC, self.sigma_min)

        # 4. Diversity restart: if sigma has collapsed and population has
        #    converged, inject fresh random individuals to escape local optima.
        #    Keep the best individual, replace the rest with perturbed copies.
        if self.sigma <= self.sigma_min * 2:
            self.sigma = min(self.sigma_max * 0.3, 1.0)
            best_ind = self.population[0].copy()
            self.population = [best_ind] + [
                best_ind + self.rng.normal(0, self.sigma, self.n)
                for _ in range(self.mu - 1)
            ]
            self.fitnesses = [self.fitnesses[0]] + [
                self.fitness_fn(ind) for ind in self.population[1:]
            ]

        # 5. Log
        best_fit = self.fitnesses[0]
        self.best_fitness_history.append(best_fit)
        self.mean_fitness_history.append(float(np.mean(self.fitnesses)))
        self.sigma_history.append(self.sigma)
        self.generation += 1

        return self.population[0].copy(), best_fit

    # ------------------------------------------------------------------ #

    def run(
        self,
        n_generations: int,
        verbose:       bool  = True,
        print_every:   int   = 10,
    ) -> Tuple[np.ndarray, float]:
        """
        Run the ES for n_generations steps.

        Returns
        -------
        best_weights : weight vector of the best individual found.
        best_fitness : its fitness value.
        """
        t0 = time.time()
        for g in range(n_generations):
            best_w, best_f = self.step()
            if verbose and (g + 1) % print_every == 0:
                elapsed = time.time() - t0
                print(
                    f"  gen {g+1:4d}/{n_generations} | "
                    f"best={best_f:8.4f} | "
                    f"mean={self.mean_fitness_history[-1]:8.4f} | "
                    f"sigma={self.sigma:.4f} | "
                    f"elapsed={elapsed:.1f}s"
                )
        return self.population[0].copy(), self.fitnesses[0]


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark(
    name:          str,
    fn:            Callable[[np.ndarray], float],
    n_dims:        int   = 10,
    n_generations: int   = 200,
    mu:            int   = 15,
    lam:           int   = 75,
    seed:          int   = 0,
) -> dict:
    """
    Minimise a benchmark function by negating it (ES maximises).

    Population is initialised uniformly over the standard search range
    for each function so the ES explores the full landscape from gen 0:
      Rastrigin  x_i in [-5.12, 5.12]   sigma_init = 2.0
      Rosenbrock x_i in [-2.048, 2.048] sigma_init = 1.0

    Returns a dict with keys:
        name, best_value, best_x, history (best per generation)
    """
    print(f"\n── Benchmark: {name} (dim={n_dims}) ──")

    is_rastrigin = "rastrigin" in name.lower()
    sigma_init   = 2.0 if is_rastrigin else 1.0
    lo, hi       = (-5.12, 5.12) if is_rastrigin else (-2.048, 2.048)

    def neg_fn(x: np.ndarray) -> float:
        return -fn(x)

    es = EvolutionStrategy(
        fitness_fn=neg_fn,
        n_weights=n_dims,
        mu=mu,
        lam=lam,
        sigma_init=sigma_init,
        sigma_min=1e-5,
        sigma_max=10.0,
        seed=seed,
    )

    # Replace the default (normal) initial population with a uniform one
    rng = np.random.default_rng(seed + 99)
    es.population = [rng.uniform(lo, hi, n_dims) for _ in range(mu)]
    es.fitnesses  = [neg_fn(ind) for ind in es.population]
    es.best_fitness_history = [max(es.fitnesses)]
    es.mean_fitness_history = [float(np.mean(es.fitnesses))]

    best_w, best_neg_f = es.run(n_generations, verbose=True, print_every=50)
    best_value = -best_neg_f

    print(f"  → best value found: {best_value:.6f}  (true minimum = 0)")

    return {
        "name":       name,
        "best_value": best_value,
        "best_x":     best_w,
        "history":    [-v for v in es.best_fitness_history],
    }


def plot_benchmarks(results: list, save_path: str = "benchmark_results.png"):
    """
    Plot convergence curves for all benchmarks and save to PNG.

    Parameters
    ----------
    results   : list of dicts returned by run_benchmark().
    save_path : output file path.
    """
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    colours = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]

    for ax, res, col in zip(axes, results, colours):
        gens = list(range(len(res["history"])))
        ax.plot(gens, res["history"], color=col, linewidth=1.8)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, label="global min = 0")
        ax.set_title(f"{res['name']}  (best = {res['best_value']:.4f})", fontsize=13)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best function value (lower = better)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("ES Benchmark Convergence — Rastrigin & Rosenbrock", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nBenchmark plot saved to: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point — run benchmarks standalone
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = [
        run_benchmark("Rastrigin",  rastrigin,  n_dims=10, n_generations=300),
        run_benchmark("Rosenbrock", rosenbrock, n_dims=10, n_generations=300),
    ]
    plot_benchmarks(results, save_path="benchmark_results.png")
    print("\nDone.  Check benchmark_results.png")