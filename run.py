print("import numpy as np

# Define MDP parameters
num_states = 3
num_actions = 2

# Define transition probabilities: P[s, a, s'] = P(s' | s, a)
transition_probs = np.array([[[0.7, 0.3, 0.0],  # From state 0, action 0
                              [1.0, 0.0, 0.0]],  # From state 0, action 1
                             [[0.0, 1.0, 0.0],  # From state 1, action 0
                              [0.8, 0.2, 0.0]],  # From state 1, action 1
                             [[0.0, 0.0, 1.0],  # From state 2, action 0
                              [0.0, 0.0, 1.0]]]) # From state 2, action 1

# Define rewards: R[s, a, s']
rewards = np.array([[[10, 0, 0],  # From state 0, action 0
                      [0, 0, 0]],  # From state 0, action 1
                     [[0, 0, 0],  # From state 1, action 0
                      [0, 0, 0]],  # From state 1, action 1
                     [[0, 0, -50],  # From state 2, action 0
                      [0, 0, 0]]])  # From state 2, action 1

# Policy Iteration
def policy_iteration(transition_probs, rewards, gamma=0.99, tol=1e-6):
    num_states, num_actions, _ = transition_probs.shape

    # Initialize random policy
    policy = np.random.choice(num_actions, size=num_states)

    while True:
        # Policy Evaluation
        values = np.zeros(num_states)
        while True:
            delta = 0
            for s in range(num_states):
                v = values[s]
                values[s] = sum([transition_probs[s, policy[s], s1] * (rewards[s, policy[s], s1] + gamma * values[s1]) for s1 in range(num_states)])
                delta = max(delta, abs(v - values[s]))
            if delta < tol:
                break

        # Policy Improvement
        policy_stable = True
        for s in range(num_states):
            old_action = policy[s]
            policy[s] = np.argmax([sum([transition_probs[s, a, s1] * (rewards[s, a, s1] + gamma * values[s1]) for s1 in range(num_states)]) for a in range(num_actions)])
            if old_action != policy[s]:
                policy_stable = False

        if policy_stable:
            break

    return policy, values

# Value Iteration
def value_iteration(transition_probs, rewards, gamma=0.99, tol=1e-6):
    num_states, num_actions, _ = transition_probs.shape

    # Initialize values arbitrarily
    values = np.zeros(num_states)

    while True:
        delta = 0
        for s in range(num_states):
            v = values[s]
            values[s] = max([sum([transition_probs[s, a, s1] * (rewards[s, a, s1] + gamma * values[s1]) for s1 in range(num_states)]) for a in range(num_actions)])
            delta = max(delta, abs(v - values[s]))

        if delta < tol:
            break

    # Extract policy from values
    policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        policy[s] = np.argmax([sum([transition_probs[s, a, s1] * (rewards[s, a, s1] + gamma * values[s1]) for s1 in range(num_states)]) for a in range(num_actions)])

    return policy, values

# Run Policy Iteration
policy_pi, values_pi = policy_iteration(transition_probs, rewards)
print("Policy (Policy Iteration):", policy_pi)
print("Values (Policy Iteration):", values_pi)

# Run Value Iteration
policy_vi, values_vi = value_iteration(transition_probs, rewards)
print("Policy (Value Iteration):", policy_vi)
print("Values (Value Iteration):", values_vi)")
