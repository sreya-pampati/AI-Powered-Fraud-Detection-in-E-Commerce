import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------------
# 1. Load fraud scores
# ----------------------------
scores = np.load('fraud_scores_per_sequence.npy')  # shape: [num_sequences, 16]
threshold = 0.3

# ----------------------------
# 2. Find top 3 sequences with most fraud spikes
# ----------------------------
fraud_flags = scores >= threshold
fraud_counts = np.sum(fraud_flags, axis=1)  # total spikes per sequence

top3_indices = np.argsort(fraud_counts)[-3:][::-1]  # Top 3 with most spikes

# ----------------------------
# 3. Plot subplots for top 3
# ----------------------------
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

for i, idx in enumerate(top3_indices):
    seq_scores = scores[idx]
    fraud_steps = np.where(seq_scores >= threshold)[0]

    axs[i].plot(range(1, 17), seq_scores, marker='o', label='Fraud Score')
    axs[i].axhline(threshold, color='red', linestyle='--', label='Threshold (0.3)')
    axs[i].scatter(fraud_steps + 1, seq_scores[fraud_steps], color='darkred', s=80, label='Fraud Spikes')

    axs[i].set_title(f"Top {i+1} Fraud Spikes – Sequence #{idx + 1} (Count: {fraud_counts[idx]})")
    axs[i].set_ylabel("Fraud Score")
    axs[i].set_ylim(0, 1.05)
    axs[i].grid(True)
    axs[i].legend()

axs[2].set_xlabel("Time Step")
plt.suptitle("📊 Top 3 Sequences with Most Frequent Fraud Spikes", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig("top3_frequent_fraud_sequences.png")
plt.close()

print("✅ Combined subplot saved as: top3_frequent_fraud_sequences.png")


# ----------------------------
# Save the figure in static folder
# ----------------------------
output_path = os.path.join('static', 'top3_frequent_fraud_sequences.png')
plt.savefig(output_path)
plt.close()

print(f"✅ Combined subplot saved as: {output_path}")

