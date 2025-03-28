# examples/custom_packages.py
import modal
import os

try:
    from modal_tf_builder import setup_modal_tf_gpu
except ImportError:
    print("Error: Could not import setup_modal_tf_gpu.")
    print("Ensure you are running this script from the root of the 'modal-tf-gpu-builder' repository,")
    print("or that the 'modal_tf_builder' package is installed/accessible in your Python path.")
    exit()

# --- Configuration ---
TARGET_PYTHON_VERSION = "3.11"
GPU_TYPE = "T4"

builder_config = {
    "app_name": "custom-pkg-example",
    "python_version": TARGET_PYTHON_VERSION,
    "gpu_type": GPU_TYPE,
    # Add extra packages needed for this specific task
    "add_pip": ["requests", "beautifulsoup4"],
    "add_mamba": {"pytorch": None} # Example: Add PyTorch alongside TF
}

# --- Setup Modal App ---
print("Setting up Modal app with custom packages...")
setup_data = setup_modal_tf_gpu(**builder_config)
app = setup_data["app"]
print(f"Modal app '{app.name}' configured.")

# --- Define Modal Function ---
@app.function(gpu=GPU_TYPE)
def task_with_custom_packages():
    import tensorflow as tf
    import torch # Check if PyTorch is available
    import requests
    from bs4 import BeautifulSoup

    print(f"TensorFlow Version: {tf.__version__}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"GPU available for PyTorch: {torch.cuda.is_available()}")

    # Use requests and BeautifulSoup
    try:
        response = requests.get("https://modal.com", timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('title').string
        print(f"Fetched title from modal.com: '{title}'")
        fetch_success = True
    except Exception as e:
        print(f"Failed to fetch or parse modal.com: {e}")
        fetch_success = False

    return {"pytorch_available": True, "web_fetch_success": fetch_success}

# --- Local Entrypoint ---
@app.local_entrypoint()
def main():
    print(f"Running task with custom packages on Modal (App: {app.name})...")
    result = task_with_custom_packages.remote()
    print(f"Remote task finished. Result: {result}")
