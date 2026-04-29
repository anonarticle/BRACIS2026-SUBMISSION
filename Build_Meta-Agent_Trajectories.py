
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import itertools
import os
from typing import Dict, Tuple, List, Any, Optional, Protocol, get_args, get_origin
import math
import time
import random
import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np

from Q_learning_Agent_plus_plus import q_learning, get_trajectory, doSeed, sizes
from MetaAgent2NIGSelfSup import MetaAgent2NIGSelfSup as MetaAgent
from MetaAgent2NIGSelfSup import value_mixture_action_temperature
from GridWorldCanonicalBenchmarkSuite import build_canonical_suite

State = Tuple[int, int]          # (x, y)
Action = int                     # 0,1,2,3

# Convention for actions:
# 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
# ACTION_DELTAS: Dict[Action, Tuple[int, int]] = {
#     0: (0, -1),
#     1: (1, 0),
#     2: (0, 1),
#     3: (-1, 0),
# }

ACTION_DELTAS = [(0,-1),(1,0),(0,1),(-1,0)]

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def softmax(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]

def normalize(probs: List[float], eps: float = 1e-12) -> List[float]:
    s = sum(probs)
    if s < eps:
        # fallback to uniform if degenerate
        return [1.0 / len(probs)] * len(probs)
    return [p / s for p in probs]


class Environment(Protocol):
    """
    Minimal interface expected by MetaAgent.
    You can adapt your own Environment class to this.
    """
    width: int
    height: int
    start: State
    max_steps: int

    def reset(self) -> State: ...
    # def step(self, action: Action) -> Tuple[State, float, bool, Dict[str, Any]]: ...
    def step(self, action: Action) -> Tuple[State, float, bool]: ...
    def step_from(self, s: State, action: Action) -> Tuple[State, float, bool]: ...


class GridWorldEnv:
    """
    Simple reference implementation (optional). Obstacles are impassable.
    Reward: -1 for successful move to empty cell,
            -10 for attempting move into obstacle (no move),
            0 for reaching goal (customizable).
    """
    def __init__(
        self,
        width: int,
        height: int,
        obstacles: set[State],
        start_state: State,
        goal_state: State,
        max_steps: int = 200,
        step_cost: float = -1.0,
        obstacle_cost: float = -10.0,
        goal_reward: float = 0.0,
        border_end: bool = False,
    ):
        self.name = "Env"        
        self.width = width
        self.height = height
        self.walls = set(obstacles)
        self.start = start_state
        self.goal = goal_state
        self.max_steps = max_steps
        self.step_cost = step_cost
        self.obstacle_cost = obstacle_cost
        self.goal_reward = goal_reward
        self.border_end = border_end

        # actions: 0=up, 1=right, 2=down, 3=left  (0=↑,1=→,2=↓,3=←)
        self.action_deltas = ACTION_DELTAS # [(0,-1),(1,0),(0,1),(-1,0)]
        # self.action_arrows = {0:'↑',1:'→',2:'↓',3:'←'}
        self.action_arrows = {'up':'↑','right':'→','down':'↓','left':'←'}
        self.action_size = len(self.action_deltas)
        self.action_space = list(range(self.action_size))

        self._s = start_state
        self._t = 0

    def update_r_wall(self, wall_reward_like_cliff):
        # self.r_wall = self.r_cliff if wall_reward_like_cliff else self.r_step
        pass

    def reset(self, random_start=False, start_state=None) -> State:
        if start_state is None:
            if random_start:
                while True:
                    x = random.randint(0,self.width-1)
                    y = random.randint(0,self.height-1)
                    s = (x, y)
                    if s != self.goal and s not in self.walls:
                        break
                self._s = (x, y)
            else:
                self._s = self.start
        else:
            self._s = start_state

        self._t = 0
        return self._s
    
    def forbidden(self, s: State):
        return s in self.walls
                     
    def step(self, action: Action) -> Tuple[State, float, bool]: # , Dict[str, Any]]:
        """
        Deterministic transition: from 'self.state' take 'action', 
        return (next_state, reward, done).
        """
        return self.step_from(self._s, action)

    def step_from(self, s, action: Action) -> Tuple[State, float, bool]: # , Dict[str, Any]]:
        """
        Deterministic transition: from 'state' take 'action', 
        return (next_state, reward, done).
        """
        self._t += 1
        self._s = s
        x, y = self._s
        dx, dy = ACTION_DELTAS[action]
        # nx = clamp(x + dx, 0, self.width - 1)
        # ny = clamp(y + dy, 0, self.height - 1)
        # ns = (nx, ny)

        if (0 <= x + dx < self.width and
            0 <= y + dy < self.height):
            ns = (x + dx, y + dy)
            r = self.step_cost # obstacle_cost
        else:
            ns = (x, y)
            r = self.step_cost
            done = self.border_end

        info: Dict[str, Any] = {"attempted_next_state": ns}

        # Hit obstacle -> no move
        if ns in self.walls:
            ns = self._s
            r = self.obstacle_cost
            done = False
            self._s = ns
            if self._t >= self.max_steps:
                done = True
            return ns, r, done # , info

        # Move to empty cell
        self._s = ns

        # Goal?
        if ns == self.goal:
            r = self.goal_reward
            done = True
        else:            
            done = False

        if self._t >= self.max_steps:
            done = True

        return ns, r, done # , info

    def all_states(self):
        """List of all (i,j) states."""
        return [(i, j) for i in range(self.width) for j in range(self.height)]

    def all_actions(self):
        """List of action indices."""
        return list(range(len(self.action_deltas)))
    
    def print_grid(self, trajectory=None, print_rewards=None):
        """
        Print the grid blueprint: S for start, G for goal, · for empty.
        """
        print(self.name)
        nsteps = 0 if trajectory is None else len(trajectory)
        for j in range(self.height):
            row = []
            for i in range(self.width):
                if (i, j) == self.start:
                    row.append("S")
                elif (i, j) == self.goal:
                    row.append("G")
                elif (i, j) in self.walls:
                    row.append("X")
                else:
                    if trajectory is not None and (i, j) in trajectory:
                        pos = trajectory[::-1].index((i, j))
                        pos = nsteps - pos - 1
                        if pos < nsteps-1:
                            ni, nj = trajectory[pos+1]
                        else:
                            if trajectory[-1] == self.goal:
                                ni, nj = self.goal
                            else:
                                ni, nj = i, j
                        # print( ni, nj)
                        if ni > i:
                            row.append(self.action_arrows['right'])        
                        elif ni < i:
                            row.append(self.action_arrows['left'])        
                        elif nj < j:
                            row.append(self.action_arrows['up'])        
                        elif nj > j:
                            row.append(self.action_arrows['down'])   
                        else:
                            row.append("·")
                    else:
                        row.append("·")
            print(" ".join(row))
        print()  # blank line for spacing

def load_qtable(agent_name: str, env_name: str, data_dir: str, state_len: int, default_state_len: int , suffix = "") -> pd.DataFrame:
    """
    Load trajectories for a given (agent, env).

    Assumes filenames like:
        f"{agent_name}_{env_name}_trajectory.csv"
    e.g. "sarsa_10x4_trajectory.csv"

    Expected columns: episode, t, obs, act, rew
    """

    fname = f"{agent_name}_{env_name}{('-'+str(state_len)) if state_len != default_state_len else ''}_qtable{suffix}.csv"
    fpath = os.path.join(data_dir, fname)
    Q_df = pd.read_csv(fpath, index_col=0, header=[0])

    state_cols = list(Q_df.columns)[: -len(ACTION_DELTAS)]
    Q_df['state'] = Q_df.apply(lambda x: tuple([int(x[c]) for c in state_cols]), axis=1)

    Q_df = Q_df.set_index('state', drop=True).drop(columns=state_cols)

    Q_df.columns = [int(a) for a in list(Q_df.columns)]

    action_space = [a for a in sorted(list(Q_df.columns))]

    Q_df['actions'] = Q_df.apply(lambda x: np.array([x[a] for a in action_space ]), axis=1)

    Q = defaultdict(lambda: np.zeros(len(action_space)), Q_df.drop(columns=action_space).to_dict()['actions'])

    return Q, action_space



# --- 4) Extract greedy policy and best trajectory ---
def extract_greedy_policy(env, Q):
    """Derive deterministic policy π(s)=argmax_a Q(s,a)."""
    policy = {}
    for s in env.all_states():
        if s == env.goal:
            policy[s] = None
        else:
            # policy[s] = int(np.argmax(Q[s]))
            # if np.std(Q[s]) == 0:
            #     policy[s] = random.randint(0, len(Q[s])-1)
            # else:
                q_vals = Q[s]
                # policy[s] = int(np.argmax(q_vals))
                max_val = np.max(q_vals)
                # break ties randomly
                # best_actions = [a for a, q in enumerate(q_vals) if q == max_val]
                best_actions = np.flatnonzero(np.isclose(q_vals, max_val)).tolist()
                policy[s] = random.choice(best_actions)                
    return policy

def softmax_row(z, temperature=1.0):
    z = z / max(temperature, 1e-6) # / np.sum(z)
    m = np.max(z)
    e = np.exp(np.clip(z - m, -50, 50))
    return e / (np.sum(e) + 1e-12)

def extract_stochastic_policy(env, Q, temperature=1.0):
    """Derive deterministic policy π(s)=softmax(Q(s))."""
    policy = {}
    for s in env.all_states():
        if s == env.goal or env.forbidden(s) or s not in Q:
            policy[s] = None
        else:
            # policy[s] = int(np.argmax(Q[s]))
            q_vals = Q[s]
            if np.std(q_vals) == 0:
                policy[s] = np.array([1/env.action_size]*env.action_size)
            else:
                policy[s] = softmax_row(q_vals, temperature)            
    return policy

def get_trajectory(env, policy, ep=-1, greedy=False, start_state=None, max_steps=100):
    """Follow the policy from start to goal, recording states."""
    state = env.reset(start_state=start_state)  
    
    border_end0 = env.border_end  
    # env.border_end = greedy

    traj = [(state[0], state[1])]
    rows = []
    for t in range(max_steps):   # cap length to avoid infinite loops
        a = policy[state] # [(state[0], state[1])]
        if a is None:
            break
        else:
            if (isinstance(a, np.ndarray)):
                if greedy:
                    max_val = np.max(a)
                    # break ties randomly
                    # best_actions = [a for a, q in enumerate(q_vals) if q == max_val]
                    best_actions = np.flatnonzero(np.isclose(a, max_val)).tolist()
                    a = random.choice(best_actions)                                      
                else:
                    a = np.random.choice(env.action_size, p=a)
        state1, reward, done = env.step_from(state, a)
        traj.append((state1[0], state1[1]))

        rows.append({"episode":ep, "t":t,"obs":state,"act":a,"rew":reward})        
        state = state1

        if done:
            break

    env.border_end = border_end0
    return traj, rows

def extract_meta_agent_trajectories(env, meta, 
                                    mode="policy-mixture",  #  "policy-mixture"/"value-mixture"/"thompson"
                                    greedy= False,
                                    force_success=False, 
                                    force_small=False,
                                    force_from_start = False,
                                    normalize=False,
                                    multiple_starts_perc=0.2):
    ep = -1
    final_rows = []

    Q = meta.q_tables[-1]
    optimal_policy = extract_greedy_policy(env, Q)
    best_traj2, _ = get_trajectory(env, optimal_policy, max_steps=2*(env.width*env.height), start_state=env.start)

    tot_start_states = int(env.width*env.height*multiple_starts_perc)
    max_tries = max(10, int(0.2*tot_start_states))

    if force_from_start:
        start_states = [env.start] * tot_start_states
    else:
        start_states = env.all_states()
    
    for s in tqdm(start_states):
        if not env.forbidden(s):
            ep += 1
            try_num = 1

            while try_num <= max_tries:
                # Trace out the best trajectory from start to goal
                row , best_traj = meta.run_episode(env, 
                                                  acting_mode=mode, #  "value-mixture"/"thompson"
                                                  episode=ep, 
                                                  start_state=s, 
                                                  max_steps=5*(env.width+env.height) if force_small else env.width*env.height*5, 
                                                  greedy=greedy,
                                                  normalize=normalize)

                if best_traj[-1] == env.goal or not force_success: # or greedy: @@@@@@@@@@@@@
                    final_rows += row
                    break
                else:
                    try_num += 1

    df=pd.DataFrame(final_rows)

    return df


def build_meta_agent_trajectories(envs_dic, meta, 
                                  mode="value-mixture",  #  "value-mixture"/"thompson"
                                  greedy= False,                                  
                                  data_dir="./", 
                                  force_success=False, 
                                  force_small=False, 
                                  force_save=False,
                                  force_from_start = False,
                                  normalize=False,
                                  multiple_starts_perc=0.2):

    for (env, _) in envs_dic.values():

        print(f'\nBuilding Meta Agent trajectories in Env: {env.name}!')
    
        env_name = env.name # f'{env.width}x{env.height}'
        # meta = env_meta_agents[env_name]
        
        # mode_str = "pm" if mode == "policy-mixture" else "vm" if mode == "value-mixture" else "th" if mode == "thompson" else "invalid_mode"
        # demo_path2=data_dir+f"/meta_agent_{env.width}x{env.height}_learned{'_succ' if force_success else ''}{'_small' if force_small else ''}{'_greedy' if greedy else ''}_traj_{tot_exp}_{mode_str}.csv"

        # num_learned_str = '_'+str(num_trajectories) if use_multiple_trajectories else '_learned' if use_learned_trajectories else ''
        succ_str = '_succ' if force_success else ''
        small_str = '_small' if force_small else ''
        greedy_str = '_greedy' if greedy else f'_stoch{value_mixture_action_temperature:.1f}'
        mode_str = "pm" if mode == "policy-mixture" else "vm" if mode == "value-mixture" else "th" if mode == "thompson" else "invalid_mode"
        if mode_str == 'vm':
            mode_str += f'{value_mixture_action_temperature}' + ('n' if normalize_q_values_in_value_mixture else '')      

        fname = f"meta_agent_{env_name}{succ_str}{small_str}{greedy_str}_traj_{mode_str}.csv"    

        demo_path2 = os.path.join(data_dir, fname)
        
        if force_save or not os.path.exists(demo_path2):
                
            df2 = pd.DataFrame(columns=["dummy"], index=[0, 1])
            df2.to_csv(demo_path2,index=False)
                            
            df2 = extract_meta_agent_trajectories(env, meta, 
                                                mode=mode,  #  "policy-mixture"/"value-mixture"/"thompson"
                                                greedy=greedy,                                              
                                                force_success=force_success, 
                                                force_small=force_small,
                                                force_from_start=force_from_start,
                                                normalize=normalize,
                                                multiple_starts_perc=multiple_starts_perc)
            
            print(f'Saving Meta Agent trajectories to: {demo_path2}!')
            df2.to_csv(demo_path2,index=False)

def euclidean_distance(tup1, tup2):
    # Assumes tuples have the same number of elements
    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(tup1, tup2)]))
    return distance

def manhattan_distance(tup1, tup2):
    # Assumes tuples have the same number of elements
    distance = sum([abs(a - b) for a, b in zip(tup1, tup2)])
    return distance

def stateQTableIsStable(env, q, 
                        num_episodes=1000, 
                        alpha=0.1,
                        gamma=0.99,
                        epsilon_start=0.1,
                        epsilon_end=0.05,
                        start_state=None,
                        max_steps=-1,
                        count_actions=False,
                        visited_states=None,
                        skip_threshold=0.005,
                        use_tqdm=True):
    pass
   
    q, df, max_td_error_at_start = q_learning(env,
        num_episodes=min(200, num_episodes // 10), 
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        prev_Q=q,
        start_state=start_state,
        max_steps=10*(env.width*env.height),
        count_actions=count_actions,
        visited_states=visited_states,
        use_tqdm=use_tqdm) 
                
    return max_td_error_at_start < skip_threshold, max_td_error_at_start

    
# --------------------- Example usage ---------------------
if __name__ == "__main__":

    wall_reward_like_cliff = True
    multiple_starts_perc = 0.2

    test_visited_states = False
    
    count_actions_to_break_loops = True

    save_full_state_info = True

    force_save = False # True
    ma_force_successfull_trajectories = True
    force_small_trajectories = False # True
    extract_trajectories_from_start_only = True

    normalize_q_values_in_value_mixture = True    

    sort_states_reversed = True # False # 

    episode_mult = 1 # 2 # 
    
    seed = None
    # seed = 2345467781
    seed = doSeed(seed)    

    num_percs = 20
    perc0 = 0.0
    percN = 1.0      
           
    random_maze_only = True # False # 
    lambda_cliff = 100.0

    all_sizes = sizes        
    
    random_maze_str = f'_rmaze{lambda_cliff:.1f}'

    suite = build_canonical_suite(sizes, random_maze_only=random_maze_only, lambda_cliff=lambda_cliff, seed=123,
                                    num_percs=num_percs, perc0=perc0, percN=percN)

    envs_eps = [ (env, n_episodes) for (fam, env_name, i), (env, n_episodes) in suite.items() ]

    envs, num_episodes_list = zip(*envs_eps)
    envs_dic = { env_name: env for (fam, env_name, i), env in suite.items() }

    N = len(envs)//len(sizes) 

    for i, env in enumerate(envs):
        env.update_r_wall(wall_reward_like_cliff)
        env.print_grid(print_rewards=False)

    random_maze_str += f"x{episode_mult}" if episode_mult > 1 else ""

    out_sub_dir = "/traj_self_superv"
    out_sub_dir += random_maze_str + f"-N={N}"
    if not os.path.exists(f"./logs{out_sub_dir}"):
        os.makedirs(f"./logs{out_sub_dir}")
    traj_sub_dir = f"./logs{out_sub_dir}"

    qtable_sub_dir = f"/qtable_self_superv"
    qtable_sub_dir += random_maze_str  + f"-N={N}"   
    if not os.path.exists(f"./logs{qtable_sub_dir}"):
        os.makedirs(f"./logs{qtable_sub_dir}")

    agent_name = 'qlearning'

    default_state_len = len(get_args(State))

    for mode in ["value-mixture", "thompson"]:

        for size in all_sizes[:]:
            all_action_space = set()   

            sizes = [size]
            suite = build_canonical_suite(sizes, random_maze_only=random_maze_only, lambda_cliff=lambda_cliff, seed=123,
                                            num_percs=num_percs, perc0=perc0, percN=percN)

            envs_eps = [ (env, n_episodes) for (fam, env_name, i), (env, n_episodes) in suite.items() ]

            envs, num_episodes_list = zip(*envs_eps)
            envs_dic = { env_name: env for (fam, env_name, i), env in suite.items() }


            qs = []
            for nu, env in enumerate(tqdm(envs[:])):
                state_len = len(env.reset())
                env.update_r_wall(wall_reward_like_cliff)
                q, action_space = load_qtable(agent_name, env.name, data_dir=f"./logs{qtable_sub_dir}/", 
                                                state_len=state_len, default_state_len=default_state_len, suffix="")
                qs.append(q)
                all_action_space.update(action_space)

            all_action_space = sorted(list(all_action_space))

            # Meta-agent only needs an env "template" for width/height semantics
            meta = MetaAgent(env_template=envs[0], q_tables=qs, action_space=all_action_space, epsilon=0.0, use_belief_weighted_Q=False) # , unknown_rew_range = True)

            greedy=True
            build_meta_agent_trajectories(envs_dic, meta, 
                                    mode=mode,  #  "value-mixture"/"thompson"
                                    greedy= greedy,                                  
                                    data_dir=traj_sub_dir, 
                                    force_success=ma_force_successfull_trajectories, 
                                    force_small=force_small_trajectories,
                                    force_save=force_save,
                                    force_from_start=extract_trajectories_from_start_only,
                                    normalize=normalize_q_values_in_value_mixture,
                                    multiple_starts_perc=0.2)
