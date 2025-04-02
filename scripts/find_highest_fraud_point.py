import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved fraud score array
score_path = os.path.join(os.getcwd(), 'fraud_scores_per_sequence.npy')
scores = np.load(score_path)  # shape: [num_sequences, 16]

# Find the highest score and its location
max_score = np.max(scores)
sequence_idx, time_idx = np.unravel_index(np.argmax(scores), scores.shape)

# Print info
print(f"🔍 Highest Fraud Score: {max_score:.4f}")
print(f"📍 Found in Sequence: {sequence_idx + 1}")
print(f"⏱️ At Time Step: {time_idx + 1}")

# Prepare static path
STATIC_DIR = os.path.join(os.getcwd(), 'static')
os.makedirs(STATIC_DIR, exist_ok=True)

# Plot the heatmap of all fraud scores
plt.figure(figsize=(12, 6))
sns.heatmap(scores, cmap="Reds", annot=False, cbar=True)
plt.title("Fraud Scores per Sequence and Time Step")
plt.xlabel("Time Step")
plt.ylabel("Sequence Index")
plt.scatter(time_idx + 0.5, sequence_idx + 0.5, color='blue', label='Max Score')
plt.legend(loc='upper right')

# Save the plot
plot_path = os.path.join(STATIC_DIR, 'fraud_score_heatmap.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()

print(f"📊 Heatmap saved to {plot_path}")
