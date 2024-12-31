import pandas as pd
import matplotlib.pyplot as plt
import re


# Load the CSV file
df = pd.read_csv('evaluation_results_highwayspeed_rampwaittime.csv')

# Extract episode number from model file names
def extract_episode(model_file):
    match = re.search(r'checkpoint_(\d+)_\d+\.(pkl|pth)', model_file)
    if match:
        return int(match.group(1))
    return None

df['Episode'] = df['Model File'].apply(extract_episode)

# Separate DQN and Q-learning models
dqn_df = df[df['Model Type'] == 'DQN']
ql_df = df[df['Model Type'] == 'Q-learning']

dqn_df_copy = dqn_df.copy()
ql_df_copy = ql_df.copy()

dqn_df = dqn_df[['Average Highway Speed',
       'Average Ramp Wait Time', 'Average Reward', 'Episode']].groupby('Episode').mean().reset_index()
ql_df = ql_df[['Average Highway Speed',
       'Average Ramp Wait Time', 'Average Reward', 'Episode']].groupby('Episode').mean().reset_index()

always_green_values = {
    'Average Highway Speed': 17.23771657743068,
    'Average Ramp Wait Time': 0.006203813932980595,
    'Reward': -625.8660922619057
}

# Plot the results
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

# Plot Average Highway Speed
axes[0].plot(dqn_df['Episode'], dqn_df['Average Highway Speed'], label='DQN', color='blue')
axes[0].plot(ql_df['Episode'], ql_df['Average Highway Speed'], label='Q-learning', color='orange')
axes[0].axhline(y=always_green_values['Average Highway Speed'], color='green', linestyle='--', label='Always Green')
axes[0].set_title('Average Highway Speed by Episode')
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Average Highway Speed')
axes[0].legend()

# Plot Average Ramp Wait Time
axes[1].plot(dqn_df['Episode'], dqn_df['Average Ramp Wait Time'], label='DQN', color='blue')
axes[1].plot(ql_df['Episode'], ql_df['Average Ramp Wait Time'], label='Q-learning', color='orange')
axes[1].axhline(y=always_green_values['Average Ramp Wait Time'], color='green', linestyle='--', label='Always Green')
axes[1].set_title('Average Ramp Wait Time by Episode')
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Average Ramp Wait Time')
axes[1].legend()

# Plot Average Reward
axes[2].plot(dqn_df['Episode'], dqn_df['Average Reward'], label='DQN', color='blue')
axes[2].plot(ql_df['Episode'], ql_df['Average Reward'], label='Q-learning', color='orange')
axes[2].axhline(y=always_green_values['Reward'], color='green', linestyle='--', label='Always Green')
axes[2].set_title('Average Reward by Episode')
axes[2].set_xlabel('Episode')
axes[2].set_ylabel('Average Reward')
axes[2].legend()

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('model_comparison_by_episode.png')

# Show the plot
plt.show()