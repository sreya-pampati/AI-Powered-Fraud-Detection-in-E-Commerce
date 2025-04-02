import onnxruntime as ort
import numpy as np
import os

# ----------------------------
# 1. Load the ONNX model
# ----------------------------
model_path = os.path.join(os.getcwd(), 'model', 'ccf_lstm_static.onnx')
session = ort.InferenceSession(model_path)

# ----------------------------
# 2. Display Input & Output Info
# ----------------------------
print("🔍 Inputs:")
for inp in session.get_inputs():
    print(f"  Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")

print("\n🔍 Outputs:")
for out in session.get_outputs():
    print(f"  Name: {out.name}, Shape: {out.shape}, Type: {out.type}")

# ----------------------------
# 3. Create Random Input [7, 16, 220]
# ----------------------------
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

# Simulated batch of 7 transaction sequences
sample_input = np.random.rand(*input_shape).astype(np.float32)

# ----------------------------
# 4. Run Inference
# ----------------------------
output_name = session.get_outputs()[0].name
output = session.run([output_name], {input_name: sample_input})[0]  # Shape: [7, 16, 1]

# ----------------------------
# 5. Sequence Fraud Detection
# ----------------------------
fraud_threshold = 0.3
fraud_flags = []

print("\n🚦 Fraud Detection Results per Sequence:")

for i in range(output.shape[0]):
    fraud_scores = output[i, :, 0]  # Shape: [16]
    is_fraud = any(score >= fraud_threshold for score in fraud_scores)
    fraud_flags.append(is_fraud)

    print(f"  🧾 Sequence #{i+1} → {'🚨 FRAUD' if is_fraud else '✅ SAFE'}")
    print(f"     Scores: {np.round(fraud_scores, 4)}")

# ----------------------------
# 6. Summary
# ----------------------------
print("\n✅ Summary:")
print(f"  🚨 Fraudulent sequences: {sum(fraud_flags)} out of {len(fraud_flags)}")
