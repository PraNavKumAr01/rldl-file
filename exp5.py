import random
import numpy as np
import matplotlib.pyplot as plt

states = ['start', 'end1', 'end2']
actions = ['left', 'right']
rewards = {
    ('start', 'left'): -1,
    ('start', 'right'): 2,
    ('end1', 'left'): 10,
    ('end2', 'right'): 10
}

q_table = {(state, action): 0 for state in states for action in actions}

# Q-learning algorithm
gamma = 0.9  # Discount factor
alpha = 0.1  # Learning rate
num_episodes = 1000

# Store Q-values over episodes
q_values_over_time = []

for episode in range(num_episodes):
    state = 'start'
    done = False
    
    while not done:
        # Choose action based on epsilon-greedy policy
        if random.random() < 0.1:
            action = random.choice(actions)
        else:
            action = max(actions, key=lambda a: q_table[(state, a)])
        
        # Take action and observe new state and reward
        new_state = 'end1' if action == 'left' else 'end2'
        reward = rewards.get((state, action), 0)
        
        # Update Q-table
        q_table[(state, action)] = (1 - alpha) * q_table[(state, action)] + \
                                  alpha * (reward + gamma * max(q_table[(new_state, a)] for a in actions))
        
        state = new_state
        done = new_state in ['end1', 'end2']
    
    # Store Q-values over time
    q_values_over_time.append(list(q_table.values()))
    
    # Print training statistics
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(q_values_over_time[-100:])
        print(f'Episode {episode + 1}, Average Q-value: {avg_reward:.2f}')

# Visualize Q-values over episodes
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(num_episodes), np.mean(q_values_over_time, axis=1))
ax.set_xlabel('Episode')
ax.set_ylabel('Average Q-value')
ax.set_title('Q-values over Time')
plt.show()

# Print the final Q-table
print("\nFinal Q-table:")
for state, action in sorted(q_table):
    print(f"{state}, {action}: {q_table[(state, action)]:.2f}")