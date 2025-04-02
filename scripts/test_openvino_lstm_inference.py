from openvino.runtime import Core
import numpy as np
import os

# ----------------------------
# 1. Load OpenVINO model
# ----------------------------
ie = Core()
model_path = os.path.join(os.getcwd(), "model", "openvino_lstm", "ccf_lstm_static.xml")
model = ie.read_model(model_path)
compiled_model = ie.compile_model(model, "CPU")

# ----------------------------
# 2. Get Input/Output Info
# ----------------------------
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

print("🔍 Input Shape:", input_layer.shape)
print("🔍 Output Shape:", output_layer.shape)

# ----------------------------
# 3. Generate Random Input
# ----------------------------
input_data = np.random.rand(*input_layer.shape).astype(np.float32)

# ----------------------------
# 4. Run Inference
# ----------------------------
result = compiled_model([input_data])[output_layer]  # Shape: [7, 16, 1]

# ----------------------------
# 5. Sequence-level Fraud Detection
# ----------------------------
fraud_threshold = 0.3
fraud_flags = []

print("\n🚦 Fraud Detection per Sequence:")

for i in range(result.shape[0]):
    fraud_scores = result[i, :, 0]  # [16]
    is_fraud = any(score >= fraud_threshold for score in fraud_scores)
    fraud_flags.append(is_fraud)

    print(f"  🧾 Sequence #{i+1} → {'🚨 FRAUD' if is_fraud else '✅ SAFE'}")
    print(f"     Scores: {np.round(fraud_scores, 4)}")

# ----------------------------
# 6. Summary
# ----------------------------
print("\n✅ Summary:")
print(f"  🚨 Fraudulent sequences: {sum(fraud_flags)} out of {len(fraud_flags)}")
