# Algorithm 3-1. Q-learning (off-policy TD)
# 1: input: a policy that uses the action‐value function, π(a∣s,Q(s,a))
# 2: Initialize Q(s,a) = 0, for all s ∈ S, a ∈ A(s)
# 3: loop: for each episode
# 4: 		Initialize environment to provide s
# 5: 		do:
# 6: 			Choose a from s using π, breaking ties randomly
# 7: 			Take action, a, and observe r,s′
# 8: 			Q(s,a) = Q(s,a) + α[r + γ*argmax((a_s ∈ A(s), Q(s′,a_s))−Q(s,a)]
# 9: 			s = s′
#10: while s is not terminal

import os
import time
import numpy as np
import random
from collections import defaultdict
from typing import Dict, Tuple, List, Any, Optional, Protocol
from tqdm import tqdm 
import pandas as pd

ACTION_DELTAS = [(0,-1),(1,0),(0,1),(-1,0)]

from GridWorldCanonicalBenchmarkSuite import build_canonical_suite

# --- 2) ε-greedy behavior policy based on Q ---
def choose_action_epsilon_greedy(Q, state, nA, epsilon, action_count=None):
    """Select action using ε-greedy over Q[state]."""
    if random.random() < epsilon:
        return random.randrange(nA)
    else:
        # break ties randomly among best actions
        q_vals = Q[state]

        if action_count is not None:
            q_vals = np.array([qv*(1-np.sign(qv)*action_count[state][i]/100) for i, qv in enumerate(q_vals)])

        max_val = np.max(q_vals)
        # break ties randomly
        # best_actions = [a for a, q in enumerate(q_vals) if q == max_val]
        best_actions = np.flatnonzero(np.isclose(q_vals, max_val)).tolist()        
        action = random.choice(best_actions)

        if action_count is not None:
            action_count[state][action] += 1

        return action

def fake_tqdm(iter, desc):
    return iter

def doSeed(seed=None, verbose=True):
    if seed is None:
        seed = int(time.time()*1000) % (2**32) 
    
    if verbose: print(seed)
    np.random.seed(seed)
    random.seed(seed)

    return seed

# --- 3) Q-learning off-policy TD control ---
def q_learning(env, num_episodes, alpha=0.5, 
               gamma=1.0,  
               epsilon_start=0.1,
               epsilon_end=0.05,
               prev_Q=None, 
               start_state=None,
               max_steps=-1,
               count_actions=False,
               visited_states = None,
               use_tqdm = True
               ):
    """
    Off-policy Q-learning:
    Q(s,a) ← Q(s,a) + α [r + γ max_{a'} Q(s',a') - Q(s,a)].
    Behavior policy is ε-greedy on Q.
    """

    def eps_by_episode(ep):
        # linear decay
        frac = min(1.0, ep / max(1, epsilon_decay_steps))
        return epsilon_start + frac * (epsilon_end - epsilon_start)
    
    epsilon_decay_steps = num_episodes // 2

    nA = len(env.all_actions())    
    action_count = None

    # initialize Q(s,a)=0 for all state-action pairs
    if prev_Q is None:
        Q = defaultdict(lambda: np.zeros(nA))
    else:
        Q = prev_Q

    max_td_error_at_start = 0.0

    if use_tqdm:
        tqdm_iter = tqdm
    else:
        tqdm_iter = fake_tqdm

    rows = []
    for episode in tqdm_iter(range(num_episodes), desc="Episodes"):

        if count_actions:
            action_count=defaultdict(lambda: [0]*nA)

        epsilon = eps_by_episode(episode)

        start_state = env.reset(start_state=start_state)
        state = start_state

        done = False
        row = []
        t = -1
        while not done:
            t += 1

            if visited_states is not None:
                visited_states[state] += 1

            # --- Real interaction step ---
            action = choose_action_epsilon_greedy(Q, state, nA, epsilon, action_count)
            next_state, reward, done = env.step_from(state, action)

            # TD target uses the max over next-state actions
            best_next_q = np.max(Q[next_state])
            td_target = reward + gamma * best_next_q
            td_error = td_target - Q[state][action]

            # Q-learning update
            Q[state][action] += alpha * td_error

            row.append({"episode":episode,"t":t,"obs":state,"act":action,"rew":reward})

            state = next_state

            # Track max delta at this state
            if abs(td_error) > max_td_error_at_start and state == start_state:
                max_td_error_at_start = abs(td_error)

            if max_steps > 0 and t > max_steps:
                break

        if done:
            rows += row[-1:]
        else:
            break

    df=pd.DataFrame(rows if done else [])

    return Q, df, max_td_error_at_start


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


def extract_learned_trajectories(env, policy, greedy=False, force_success=False, force_small=False, force_from_start=False, multiple_starts_perc = 0.2):
    ep = -1
    final_rows = []

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
                best_traj, row = get_trajectory(env, policy, ep=ep, greedy=greedy, start_state=s, 
                                                            max_steps=5*(env.width+env.height) if force_small else 5*(env.width*env.height))
                if best_traj[-1] == env.goal or not force_success:
                    final_rows += row
                    break
                else:
                    try_num += 1

    df=pd.DataFrame(final_rows)

    return df

def load_qtable(agent_name: str, env: Any, data_dir: str, state_len: int, default_state_len: int) -> pd.DataFrame:
    """
    Load trajectories for a given (agent, env).

    Assumes filenames like:
        f"{agent_name}_{env_name}_trajectory.csv"
    e.g. "sarsa_10x4_trajectory.csv"

    Expected columns: episode, t, obs, act, rew
    """

    env_name = env.name
    fname = f"{agent_name}_{env_name}{('-'+str(state_len)) if state_len != default_state_len else ''}_qtable.csv"
    fpath = os.path.join(data_dir, fname)
    if os.path.exists(fpath):

        Q_df = pd.read_csv(fpath, index_col=0, header=[0])
        state_cols = list(Q_df.columns)[:-len(ACTION_DELTAS)]

        Q_df['state'] = Q_df.apply(lambda x: tuple([int(x[c]) for c in state_cols]), axis=1)

        Q_df = Q_df.set_index('state', drop=True).drop(columns=state_cols)
        
        Q_df.columns = [int(a) for a in list(Q_df.columns)]

        action_space = [a for a in sorted(list(Q_df.columns))]

        Q_df['actions'] = Q_df.apply(lambda x: np.array([x[a] for a in action_space ]), axis=1)

        Q = Q_df.drop(columns=action_space).to_dict()['actions']
    else:
        Q = None
        action_space = None

    return Q, action_space


    
def build_trajectories_from_saved_Q_table(sizes):

    num_percs = 20
    perc0 = 0.0
    percN = 1.0      
           
    random_maze_only = True 
    lambda_cliff = 100.0 

    random_maze_str = f'_rmaze{lambda_cliff:.1f}'

    suite = build_canonical_suite(sizes, random_maze_only=random_maze_only, lambda_cliff=lambda_cliff, seed=123,
                                    num_percs=num_percs, perc0=perc0, percN=percN)

    envs = { env_name: env for (fam, env_name, i), env in suite.items() }

    N = len(envs)//len(sizes)

    temperature = 0.3
 
    agent_name = "qlearning"
    default_state_len = 2

    force_successfull_trajectories = True
    force_small_trajectories = False # True
    extract_trajectories_from_start_only = True

    use_greedy_policy_to_build_trajectories_from_saved_Q_table = True # False # 

    episode_mult = 1 # 2 # 
    
    out_sub_dir = "/traj_self_superv"
    qtable_sub_dir = "/qtable_self_superv"

    random_maze_str += f"x{episode_mult}" if episode_mult > 1 else ""
    
    out_sub_dir += random_maze_str + f"-N={N}"
    qtable_sub_dir += random_maze_str + f"-N={N}"

    if not os.path.exists(f"./logs{out_sub_dir}"):
        os.makedirs(f"./logs{out_sub_dir}")

    if not os.path.exists(f"./logs{qtable_sub_dir}"):
        os.makedirs(f"./logs{qtable_sub_dir}")

    succ_small_str = ''
    succ_small_str += '_succ' if force_successfull_trajectories else ''
    succ_small_str += '_small' if force_small_trajectories else ''
    succ_small_str += ('_greedy' if use_greedy_policy_to_build_trajectories_from_saved_Q_table else 
                        f'_stoch{temperature:.1f}')

    for (env, episodes) in list(envs.values())[:]:

        print(f'Building trajectories in Env: {env.name}!\n')

        env_name = env.name # f'{env.width}x{env.height}'
        state_len = len(env.reset())
        num_actions = len(env.all_actions())

        Q, action_space  = load_qtable(agent_name, env, 
                                    data_dir=f"./logs{qtable_sub_dir}", 
                                    state_len=state_len, default_state_len=default_state_len)

        if Q is None:
            print(f'Q table not found for Env: {env.name}!\n')
        else:
            demo_path2=f"./logs{out_sub_dir}/{agent_name}_{env.name}{succ_small_str}_traj.csv"

            if os.path.exists(demo_path2):
                print(f'\n{demo_path2} already exists! Skipping to next env size.\n')
                continue   

            df2 = pd.DataFrame(columns=["dummy"], index=[0, 1])
            df2.to_csv(demo_path2,index=False)

            # Derive greedy (optimal) policy from Q
            policy = (Q if use_greedy_policy_to_build_trajectories_from_saved_Q_table else
                                    extract_stochastic_policy(env, Q, temperature))

            df2 = extract_learned_trajectories(env, policy, 
                                            greedy=use_greedy_policy_to_build_trajectories_from_saved_Q_table,
                                            force_success=force_successfull_trajectories, 
                                            force_small=force_small_trajectories,
                                            force_from_start=extract_trajectories_from_start_only)
            df2.to_csv(demo_path2,index=False)

# --- 5) Demonstration on a 4×4 grid ---
if __name__ == "__main__":
    OUTPUT_DIR = './logs/'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    all_sizes = [(10, 5), (20, 10), (40, 20), (60,30), (80, 40), (100, 50)]

    sizes = [(10, 5), (20, 10), (40, 20), (60,30), (80, 40), (100, 50)]
    sizes = [(20, 10), (40, 20), (60,30), (80, 40), (100, 50)]
    sizes = [(40, 20), (60,30), (80, 40), (100, 50)]
    sizes = [(60,30), (80, 40), (100, 50)]
    sizes = [(80, 40), (100, 50)]
    sizes = [(100, 50)]
    # sizes = [(10, 5)]
    sizes = [(20, 10)]
    # sizes = [(40, 20)]
    # sizes = [(60,30)]
    # sizes = [(80, 40)]
    # sizes = [(100, 50)]     

    build_trajectories_from_saved_Q_table(sizes)
