In order to reproduce article results, please perform the following steps in the specified order.

1. Train Q-agents with Train_Q_Agents.py
2. Build Q-agents trajectories with Q_learning_Agent_plus_plus.py
3. Build Meta-agents trajectories with Build_Meta-Agent_Trajectories.py 
4. Run MultiEnv_gridworld_online_mixture_of_experts.ipynb multiple times, once for each grid size!
5. Calc Navigation Coherence score with navigational_coherence_new.py
6. Test generalization bound with experiment_b_generalization_bound_v2.py


In order to change which grid sizes to use, go to the end of file Q_learning_Agent_plus_plus.py and choose the desired grid sizes.

