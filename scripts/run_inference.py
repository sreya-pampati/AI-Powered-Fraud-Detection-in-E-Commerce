from openvino.runtime import Core
import numpy as np

# Load OpenVINO model
ie = Core()
model_path = "model\lightgbm_fraud_detection.onnx"  # Or IR .xml later
model = ie.read_model(model_path)
compiled_model = ie.compile_model(model=model, device_name="CPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Simulated sample transaction (replace with actual input shape)
transaction = np.array([[0.5, 1.2, 0.0, 0.33, 0.8]]).astype(np.float32)

# Inference
output = compiled_model([transaction])[output_layer]
print("Fraud Probability:", output[0][0])
