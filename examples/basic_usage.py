# examples/basic_usage.py
import modal
import os
from modal_tf_builder import setup_modal_tf_gpu

# --- Configuration ---
GPU_TYPE = "T4" # Or "A10G", etc.

builder_config = {
    "app_name": "basic-tf-example",
    "gpu_type": GPU_TYPE,
    # Add any extra packages needed specifically for this example
    # "add_pip": ["some_plotting_lib"],
}

# --- Setup Modal App ---
print("Setting up Modal app...")
setup_data = setup_modal_tf_gpu(**builder_config)
app = setup_data["app"]
print(f"Modal app '{app.name}' configured.")

# --- Define Modal Function ---
@app.function(gpu=GPU_TYPE)
def simple_tf_task():
    import tensorflow as tf
    import numpy as np
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

    # Perform a simple calculation
    a = tf.constant(np.arange(10), dtype=tf.float32)
    b = tf.constant(np.arange(10, 20), dtype=tf.float32)
    c = tf.add(a, b)
    print("Calculation result (first 5 elements):", c.numpy()[:5])
    return {"result_sum_first_element": float(c.numpy()[0])}

# --- Local Entrypoint ---
@app.local_entrypoint()
def main():
    print(f"Running basic TF task on Modal (App: {app.name})...")
    result = simple_tf_task.remote()
    print(f"Remote task finished. Result: {result}")
