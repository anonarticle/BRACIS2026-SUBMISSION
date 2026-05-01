"""
Experiment C — Option 3 (corrected): Phase Transition via In-F* Perturbation.

What changed from v1
--------------------
v1 built a separate test maze for each rho_test by calling build_canonical_suite
again at that density.  When rho_test < max(family_densities), the test maze
had MORE free cells than the family's F*, so the path search domain was
larger than F*, Q_mix was undefined on cells in (test_free \\ F*), and the
artifact dominated every result with rho_test < max(family).

v2 fixes this by treating ρ_test as a TEST PERTURBATION rate within F*,
not as a redefinition of the maze:

  • Family is the full 20-Q-table set, fixed throughout.
  • F* is computed once.  Q_mix is pre-cached on F*.  Both are constants.
  • For each ρ_test ∈ [0, ρ_max], sample a random subset of F* cells of
    size ρ_test · |F*_interior| as "additional obstacles."
  • Find the coherence-maximising widest-path through F* MINUS the masked
    cells.  Record c_score and ε_margin.
  • Repeat M times per ρ_test with different random masks → get a
    distribution of (c_score, ε_margin) at each test density.

Now ρ_test parametrises test load (how much extra obstacle pressure the
path must navigate around), not the maze identity.  Q_mix is defined on
every cell the path visits, so c_score reflects actual coherence rather
than F*-boundary artifacts.

Theory tested
-------------
Theorem 7.1: ε_margin transitions discontinuously at ρ_test = ρ_c, OR
has a continuous band whose width determines the value of the
ε-relaxation.  By sampling M masks per ρ_test, we observe the
*distribution* of outcomes at each density — a sharp transition shows
as a bimodal jump in the distribution; a continuous band shows as a
gradually shifting unimodal distribution.

Output
------
  - CSV: per-(ρ_test, sample) row with margin statistics.
  - Plot: coherence and ε_margin vs ρ_test, with per-density
    distribution clouds.

Usage
-----
    python experiment_c_phase_transition_v2.py
"""

import os
import sys
import math
import random
import heapq
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from navigational_coherence_new import (
    _identify_consensus_free_space,
    _compute_q_mix,
    ACTION_DELTAS,
    State,
)
from GridWorldCanonicalBenchmarkSuite import build_canonical_suite


def load_qtable(agent_name: str, env_name: str, data_dir: str,
                state_len: int, default_state_len: int,
                suffix: str = "") -> Tuple[defaultdict, list]:
    fname = (
        f"{agent_name}_{env_name}"
        f"{('-' + str(state_len)) if state_len != default_state_len else ''}"
        f"_qtable{suffix}.csv"
    )
    fpath = os.path.join(data_dir, fname)
    Q_df = pd.read_csv(fpath, index_col=0, header=[0])
    state_cols = list(Q_df.columns)[: -len(ACTION_DELTAS)]
    Q_df['state'] = Q_df.apply(
        lambda x: tuple([int(x[c]) for c in state_cols]), axis=1)
    Q_df = Q_df.set_index('state', drop=True).drop(columns=state_cols)
    Q_df.columns = [int(a) for a in list(Q_df.columns)]
    action_space = sorted(list(Q_df.columns))
    Q_df['actions'] = Q_df.apply(
        lambda x: np.array([x[a] for a in action_space]), axis=1)
    Q = defaultdict(
        lambda: np.zeros(len(action_space)),
        Q_df.drop(columns=action_space).to_dict()['actions'],
    )
    return Q, action_space


# =====================================================================
# 1.  Coherence-maximising widest-path on a given allowed-cell set
# =====================================================================

def _normalised_margin(q: np.ndarray, action: int,
                       cf_actions: List[int]) -> float:
    q_chosen = q[action]
    q_others = [q[a] for a in cf_actions if a != action]
    if not q_others:
        return 1.0
    q_best_other = max(q_others)
    all_q = [q[a] for a in cf_actions]
    q_range = max(all_q) - min(all_q)
    if q_range < 1e-12:
        return 0.0
    return (q_chosen - q_best_other) / q_range


def _consensus_free_actions_in_set(s: State, allowed: Set[State],
                                   W: int, H: int) -> List[int]:
    actions = []
    for a, (dx, dy) in enumerate(ACTION_DELTAS):
        tx = max(0, min(W - 1, s[0] + dx))
        ty = max(0, min(H - 1, s[1] + dy))
        target = (tx, ty)
        if target == s:
            continue
        if target in allowed:
            actions.append(a)
    return actions


def coherence_path_in_allowed_set(
    q_mix_cache: Dict[State, np.ndarray],
    allowed: Set[State],
    start: State,
    goal: State,
    W: int,
    H: int,
) -> Tuple[float, Optional[List[State]], List[float]]:
    """Max-min Dijkstra: path that maximises the minimum normalised margin
    along it, with cells restricted to `allowed`."""
    if start not in allowed or goal not in allowed:
        return float('-inf'), None, []

    INF = float('inf')
    heap = [(-INF, start)]
    best_bn = defaultdict(lambda: float('-inf'))
    best_bn[start] = INF
    came_from: Dict[State, Optional[Tuple[State, int]]] = {start: None}

    while heap:
        neg_bn, s = heapq.heappop(heap)
        bn = -neg_bn
        if bn < best_bn[s]:
            continue
        if s == goal:
            path: List[State] = []
            actions_taken: List[int] = []
            cur: Optional[State] = goal
            while cur is not None:
                path.append(cur)
                prev = came_from.get(cur)
                if prev is None:
                    break
                actions_taken.append(prev[1])
                cur = prev[0]
            path.reverse()
            actions_taken.reverse()
            margins: List[float] = []
            for i in range(len(path) - 1):
                ps = path[i]
                a = actions_taken[i]
                cf = _consensus_free_actions_in_set(ps, allowed, W, H)
                q = q_mix_cache.get(ps)
                if q is None:
                    margins.append(float('nan'))
                else:
                    margins.append(_normalised_margin(q, a, cf))
            return (bn if bn != INF else 1.0), path, margins

        cf_actions = _consensus_free_actions_in_set(s, allowed, W, H)
        if not cf_actions:
            continue
        q = q_mix_cache.get(s)
        if q is None:
            continue
        for a in cf_actions:
            dx, dy = ACTION_DELTAS[a]
            s_next = (max(0, min(W - 1, s[0] + dx)),
                      max(0, min(H - 1, s[1] + dy)))
            if s_next == s or s_next not in allowed:
                continue
            margin = _normalised_margin(q, a, cf_actions)
            new_bn = min(bn, margin)
            if new_bn > best_bn[s_next]:
                best_bn[s_next] = new_bn
                came_from[s_next] = (s, a)
                heapq.heappush(heap, (-new_bn, s_next))

    return float('-inf'), None, []


def epsilon_margin_from_margins(margins: List[float]) -> float:
    valid = [m for m in margins if not math.isnan(m)]
    if not valid:
        return 1.0
    return sum(1 for m in valid if m < 0) / len(valid)


# =====================================================================
# 2.  Main runner: in-F* perturbation sweep
# =====================================================================

def run_experiment_c_v2(
    grid_size: Tuple[int, int],
    family_densities: List[float],
    test_densities: List[float],
    n_samples_per_density: int = 30,
    agent_name: str = 'qlearning',
    qtable_dir_template: str = './logs/qtable_self_superv_rmaze100.0-N={N}',
    suite_seed: int = 123,
    perturbation_seed: int = 7,
    lambda_cliff: float = 100.0,
    wall_reward_like_cliff: bool = True,
    output_csv: Optional[str] = None,
    output_plot: Optional[str] = None,
    verbose: bool = True,
):
    """
    Corrected Experiment C with in-F* perturbation.

    Parameters
    ----------
    family_densities : densities of the FIXED family (use the full 20).
    test_densities   : ρ_test values to sweep.  ρ_test is the fraction
                       of F*'s interior cells to mask as additional
                       obstacles per sample.
    n_samples_per_density : M random masks drawn at each ρ_test.
                            Yields a distribution of (c_score, ε_margin).
    perturbation_seed : seed for the random-mask generator.
    """
    W, H = grid_size

    # N_family = len(family_densities)
    N_family = int(1/(family_densities[1]-family_densities[0])+0.1)
    perc0_family = min(family_densities)
    percN_family = max(family_densities)

    if verbose:
        print(f"{'='*72}")
        print(f"Experiment C v2: In-F* Perturbation Phase Transition")
        print(f"{'='*72}")
        print(f"Grid: {W}×{H}")
        print(f"Family: N = {len(family_densities)} densities in "
              f"[{min(family_densities)}, {max(family_densities)}]  (FIXED)")
        print(f"Test ρ_perturbation: {len(test_densities)} values, "
              f"{n_samples_per_density} samples each")
        print()

    # ----- Build the FIXED family of envs -----
    family_suite = build_canonical_suite(
        [grid_size],
        random_maze_only=True,
        lambda_cliff=lambda_cliff,
        seed=suite_seed,
        num_percs=N_family,
        perc0=perc0_family,
        percN=percN_family,
        verbose=False,
    )
    family_envs = [env for (_fam, _name, _i), (env, _eps)
                   in family_suite.items()]
    for env in family_envs:
        env.update_r_wall(wall_reward_like_cliff)

    # Load Q-tables
    qtable_dir = qtable_dir_template.format(N=N_family)
    if verbose:
        print(f"Loading Q-tables from: {qtable_dir}")
    q_tables = []
    for env in tqdm(family_envs, desc="Loading family Q-tables",
                    disable=not verbose):
        state_len = len(env.reset())
        Q, _ = load_qtable(agent_name, env.name, data_dir=qtable_dir,
                           state_len=state_len, default_state_len=2)
        q_tables.append(Q)

    start = family_envs[0].start
    goal = family_envs[0].goal

    # ----- Compute F* and Q_mix (ONCE, fixed throughout) -----
    f_star = _identify_consensus_free_space(
        q_tables, W, H, start=start, goal=goal,
    )
    if verbose:
        print(f"\nFixed family F*: |F*| = {len(f_star)} cells "
              f"(density {len(f_star)/(W*H):.4f})")
        print(f"Pre-caching Q_mix on F*...")
    # q_mix_cache = {s: _compute_q_mix(q_tables, s) for s in f_star}
    all_cells = {(x, y) for x in range(W) for y in range(H)}    
    q_mix_cache = {s: _compute_q_mix(q_tables, s) for s in all_cells}

    # The cells eligible for perturbation: F* minus start and goal
    # (we never mask those — must be reachable).
    perturbable = list(f_star - {start, goal})
    n_perturbable = len(perturbable)
    if verbose:
        print(f"Perturbable interior of F*: {n_perturbable} cells")

    # ----- Baseline: unperturbed coherence on F* -----
    c_base, path_base, margins_base = coherence_path_in_allowed_set(
        q_mix_cache, f_star, start, goal, W, H,
    )
    eps_base = epsilon_margin_from_margins(margins_base) if margins_base else 1.0
    if verbose:
        print(f"\nBaseline (ρ_test=0): C={c_base:.4f}  "
              f"ε_margin={eps_base:.4f}  "
              f"|P*|={len(path_base) - 1 if path_base else 0}")
        print()
        print(f"Sweeping ρ_test (in-F* perturbation rate):")

    # ----- Sweep ρ_test, sample M masks each -----
    rng = random.Random(perturbation_seed)
    rows = []

    for rho_test in tqdm(test_densities, desc="ρ_test sweep",
                         disable=not verbose):
        n_mask = int(round(rho_test * n_perturbable))
        for sample_idx in range(n_samples_per_density):
            if n_mask == 0:
                masked: Set[State] = set()
            else:
                masked = set(rng.sample(perturbable, n_mask))
            allowed = f_star - masked
            c_score, path, margins = coherence_path_in_allowed_set(
                q_mix_cache, allowed, start, goal, W, H,
            )
            if path is None:
                row = {
                    'rho_test': rho_test, 'sample': sample_idx,
                    'n_masked': n_mask, 'path_len': 0,
                    'c_score': float('nan'), 'eps_margin': 1.0,
                    'mean_margin': float('nan'), 'min_margin': float('nan'),
                    'max_margin': float('nan'), 'std_margin': float('nan'),
                    'n_negative': 0, 'verdict': 'NO PATH',
                }
            else:
                valid = [m for m in margins if not math.isnan(m)]
                eps_m = epsilon_margin_from_margins(margins)
                verdict = (
                    'FREE INFERENCE HOLDS' if c_score > 1e-9 else
                    'MARGINAL' if c_score > -1e-9 else
                    'FREE INFERENCE FAILS'
                )
                row = {
                    'rho_test': rho_test, 'sample': sample_idx,
                    'n_masked': n_mask, 'path_len': len(path) - 1,
                    'c_score': c_score, 'eps_margin': eps_m,
                    'mean_margin': float(np.mean(valid)) if valid else float('nan'),
                    'min_margin': float(np.min(valid)) if valid else float('nan'),
                    'max_margin': float(np.max(valid)) if valid else float('nan'),
                    'std_margin': float(np.std(valid)) if valid else float('nan'),
                    'n_negative': sum(1 for m in valid if m < 0),
                    'verdict': verdict,
                }
            rows.append(row)

    df = pd.DataFrame(rows)
    df['rho_max'] = df['rho_test']+family_densities[-1]

    rho_col = 'rho_test'
    # rho_col = 'rho_max'

    # ----- Aggregate: per-density mean and quantiles -----
    agg = df.groupby(rho_col).agg(
        c_mean=('c_score', 'mean'),
        c_std=('c_score', 'std'),
        c_min=('c_score', 'min'),
        c_max=('c_score', 'max'),
        eps_mean=('eps_margin', 'mean'),
        eps_std=('eps_margin', 'std'),
        eps_min=('eps_margin', 'min'),
        eps_max=('eps_margin', 'max'),
        frac_fail=('c_score', lambda v: (v < -1e-9).mean()),
        frac_marginal=('c_score', lambda v: ((v >= -1e-9) & (v <= 1e-9)).mean()),
        frac_holds=('c_score', lambda v: (v > 1e-9).mean()),
    ).reset_index()

    if output_csv is not None:
        output_csv = output_csv.replace('.csv', '-perturb' if rho_col=='rho_test' else '-density')
        df.to_csv(output_csv, index=False, sep=';', decimal=',')
        agg_csv = output_csv.replace('.csv', '-agg.csv').replace('.csv', '-perturb' if rho_col=='rho_test' else '-density')
        agg.to_csv(agg_csv, index=False, sep=';', decimal=',')
        if verbose:
            print(f"\nResults saved → {output_csv}")
            print(f"Aggregated  → {agg_csv}")

    if verbose:
        print(f"\n{'='*72}")
        print(f"AGGREGATED RESULTS")
        print(f"{'='*72}")
        # Print a clean table
        print(f"{rho_col:>8} | {'C mean':>8} | {'C std':>7} | "
              f"{'ε mean':>7} | {'ε std':>7} | "
              f"{'%hold':>5} | {'%marg':>5} | {'%fail':>5}")
        for _, r in agg.iterrows():
            print(f" {r[rho_col]:>7.4f} | "
                  f"{r['c_mean']:>8.4f} | {r['c_std']:>7.4f} | "
                  f"{r['eps_mean']:>7.4f} | {r['eps_std']:>7.4f} | "
                  f"{100*r['frac_holds']:>4.0f}% | {100*r['frac_marginal']:>4.0f}% | "
                  f"{100*r['frac_fail']:>4.0f}%")

    if output_plot is not None:
        output_plot=output_plot.replace('.png', '-perturb' if rho_col=='rho_test' else '-density')
        _plot_v2(df, agg, rho_col, family_densities[-1], grid_size, len(family_densities), 
                 n_samples_per_density, output_plot, rho_col=='rho_test')
        if verbose:
            print(f"Plot saved → {output_plot}")

    return df, agg


def _plot_v2(df: pd.DataFrame, agg: pd.DataFrame, rho_col: str, 
             max_rho: float,
             grid_size: Tuple[int, int], N_family: int,
             n_samples: int, output_path: str, is_rho_delta: bool) -> None:
    """Three-panel plot: C distribution, ε distribution, regime fractions."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    W, H = grid_size

    # Panel 1: C(V) — sample cloud + mean
    ax = axs[0]
    ax.scatter(df[rho_col], df['c_score'],
               s=10, alpha=0.35, color='C0', label='samples')
    ax.plot(agg[rho_col], agg['c_mean'],
            marker='o', linewidth=2, color='C0', label='mean')
    ax.axhline(0, color='red', linestyle='--', linewidth=0.8, alpha=0.7,
               label='FI threshold')
    if is_rho_delta:
        ax.set_xlabel(r'$\rho_{\mathrm{test}}$ (in-F* perturbation rate)', fontsize=11) 
    else:
        ax.set_xlabel(r'$\rho_{\mathrm{test}}$ (in obstacle density)', fontsize=11) 
    ax.set_ylabel(r'$\mathcal{C}(\mathcal{V})$', fontsize=11)
    ax.set_title(f'Coherence vs {"perturbation" if is_rho_delta else "density"}  ({W}×{H}, N={N_family}, '
                 f'M={n_samples} per ρ, max ρ={max_rho})', fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)

    # Panel 2: ε_margin — sample cloud + mean
    ax = axs[1]
    ax.scatter(df[rho_col], df['eps_margin'],
               s=10, alpha=0.35, color='C1', label='samples')
    ax.plot(agg[rho_col], agg['eps_mean'],
            marker='s', linewidth=2, color='C1', label='mean')
    ax.set_xlabel(r'$\rho_{\mathrm{test}}$', fontsize=11)
    ax.set_ylabel(r'$\varepsilon_{\mathrm{margin}}$', fontsize=11)
    ax.set_title(f'Margin-violation fraction vs {"perturbation" if is_rho_delta else "density"}',
                 fontsize=11)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)

    # Panel 3: Regime fractions
    ax = axs[2]
    ax.fill_between(agg[rho_col], 0, agg['frac_holds'],
                    color='C2', alpha=0.7, label='FI holds')
    ax.fill_between(agg[rho_col], agg['frac_holds'],
                    agg['frac_holds'] + agg['frac_marginal'],
                    color='C8', alpha=0.7, label='Marginal')
    ax.fill_between(agg[rho_col],
                    agg['frac_holds'] + agg['frac_marginal'], 1,
                    color='C3', alpha=0.7, label='FI fails')
    ax.set_xlabel(r'$\rho_{\mathrm{test}}$', fontsize=11)
    ax.set_ylabel('Fraction of samples', fontsize=11)
    ax.set_title(f'Regime composition vs {"perturbation" if is_rho_delta else "density"}',
                 fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc='center left')

    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# =====================================================================
# 3.  Entry point
# =====================================================================

if __name__ == '__main__':

    # ---- Configuration ----
    GRID_SIZE = (100, 50)

    # FIXED family — use the full 20 Q-tables you have
    N_FAMILY = 20
    N_TEST_DENSITIES = 40
    N = 14 # 20 # 15 # 
    for N_FAMILY in range(8, 16+1):

        FAMILY_DENSITIES = [round(0.05 * (k + 1), 3) for k in range(N_FAMILY)]
        # → [0.05, 0.10, …, 1.00]

        # Test perturbation: fine sweep over [0, 0.5] at Δρ = 0.025
        # ρ_test is the fraction of F* cells masked as extra obstacles.
        TEST_DENSITIES = [round(0.025 * k, 4) for k in range(0, N_TEST_DENSITIES+1) if round(0.025 * k, 4) + FAMILY_DENSITIES[-1] <= 1.0]
        # → [0.000, 0.025, 0.050, …, 0.500]

        N_SAMPLES = 50        # random masks per ρ_test

        out_dir = './logs'
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(
            out_dir,
            f'experiment_c_v2-{GRID_SIZE[0]}x{GRID_SIZE[1]}'
            f'-Nfam{len(FAMILY_DENSITIES)}'
            f'-Ntest{len(TEST_DENSITIES)}'
            f'-M{N_SAMPLES}.csv'
        )
        out_plot = out_csv.replace('.csv', '.png')

        qtable_dir_template = './logs/qtable_self_superv_rmaze100.0-N={N}'

        df, agg = run_experiment_c_v2(
            grid_size=GRID_SIZE,
            family_densities=FAMILY_DENSITIES,
            test_densities=TEST_DENSITIES,
            n_samples_per_density=N_SAMPLES,
            agent_name='qlearning',
            qtable_dir_template=qtable_dir_template,
            suite_seed=123,
            perturbation_seed=7,
            lambda_cliff=100.0,
            wall_reward_like_cliff=True,
            output_csv=out_csv,
            output_plot=out_plot,
            verbose=True,
        )
        print()

    print("\nDone.")
