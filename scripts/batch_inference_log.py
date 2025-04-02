from openvino.runtime import Core
import numpy as np
import pandas as pd
import os

# ----------------------------
# 1. Load OpenVINO Model
# ----------------------------
ie = Core()
model_path = os.path.join("model", "openvino_lstm", "ccf_lstm_static.xml")
model = ie.read_model(model_path)
compiled_model = ie.compile_model(model, "CPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# ----------------------------
# 2. Set Parameters
# ----------------------------
total_sequences = 105
batch_size = input_layer.shape[0]  # Usually 7
num_batches = total_sequences // batch_size

results = []
all_scores = []  # ⏱️ Store fraud score sequences
fraud_threshold = 0.3

# ----------------------------
# 3. Run Batch Inference
# ----------------------------
for _ in range(num_batches):
    input_data = np.random.rand(*input_layer.shape).astype(np.float32)
    result = compiled_model([input_data])[output_layer]  # Shape: [batch, 16, 1]

    for i in range(result.shape[0]):
        fraud_scores = result[i, :, 0]  # shape: [16]
        all_scores.append(fraud_scores)  # ⏱️ Save scores per sequence

        max_score = float(np.max(fraud_scores))
        is_fraud = max_score >= fraud_threshold

        # 🧠 GenAI-style explanation
        if max_score > 0.9:
            explanation = "⚠️ Highly suspicious behavior with sudden score spike."
        elif max_score > 0.6:
            explanation = "⚠️ Unusual activity pattern detected — possibly high-risk."
        elif max_score > 0.3:
            explanation = "⚠️ Slightly abnormal behavior, worth investigating."
        else:
            explanation = "✅ No suspicious behavior detected."

        results.append({
            "Max Fraud Score": round(max_score, 4),
            "Fraud Detected": int(is_fraud),
            "GenAI Explanation": explanation
        })

# ----------------------------
# 4. Save Outputs
# ----------------------------
# ⏺️ Save scores for time-series analysis
np.save("fraud_scores_per_sequence.npy", np.array(all_scores))

# 📄 Save batch results
df = pd.DataFrame(results)
df.to_csv("fraud_batch_results.csv", index=False)

print("✅ Batch inference completed and saved to: fraud_batch_results.csv")
print(f"📊 Total Sequences: {len(df)} | Fraudulent: {df['Fraud Detected'].sum()}")
print("💾 Saved fraud score sequences to: fraud_scores_per_sequence.npy")
