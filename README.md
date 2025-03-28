# Modal TensorFlow GPU Environment Builder

**Author:** Osvald Nigola

## Problem Solved

Setting up a reliable and reproducible environment for running TensorFlow with GPU acceleration on Modal.com can be challenging due to complex dependencies between Python versions, CUDA, cuDNN, TensorFlow, NumPy, and system libraries. This tool automates the creation of a stable Modal Image and App, simplifying the process significantly.

## Features

*   Creates stable, reproducible TF+GPU environments on Modal using Micromamba.
*   Based on a known working combination (TF 2.14, CUDA 11.8, Python 3.11).
*   Configurable Python version (though sticking to the default is recommended for stability).
*   Easily add custom `apt`, `micromamba`, and `pip` packages to the base image.
*   Allows overriding default package versions (use with caution).
*   Includes optional verification tests to confirm GPU functionality after image setup.
*   Provides both a Command Line Interface (CLI) for direct use and a Python function for integration into your Modal scripts.
*   Performs basic local environment checks.

## Prerequisites

*   **Python:** Python 3.10+ installed locally.
*   **Modal Client:** `modal-client` installed locally (`pip install modal-client`).
    *   *Crucially*, ensure your local Python **major.minor** version (e.g., 3.11) matches the Python version used for the Modal image (default is 3.11) to avoid serialization errors when passing complex objects between your local machine and Modal containers.
*   **tblib:** `tblib` installed locally (`pip install tblib`). Recommended by Modal for improved stability with tracebacks.
*   **Modal Account:** A configured Modal account (`modal token set ...`).
*   **(Optional) Conda/Micromamba:** Useful for managing your local Python environment.

## Installation / Setup

This tool is designed to be used directly from its repository directory, not typically installed globally.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Ozzuke/modal-tf-gpu-builder.git
    cd modal-tf-gpu-builder
    ```
2.  **(Recommended) Create and activate a local virtual environment:**
    *   Using `venv`:
        ```bash
        python -m venv .venv
        source .venv/bin/activate
        ```
    *   Using `conda`:
        ```bash
        conda create -n modal-builder python=3.11 # Match the default remote version
        conda activate modal-builder
        ```
3.  **Install local dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Command Line Interface (CLI)

Run the builder script directly from the terminal within the repository directory. This is useful for testing configurations or creating one-off apps.

```bash
# Test the default setup (builds image, runs tests with verbose output)
python -m modal_tf_builder.builder --run-tests --verbose-tests

# Create an app named 'my-tf-job' with additional packages, then test
python -m modal_tf_builder.builder --app-name my-tf-job \
    --add-pip "requests>=2.28" "pandas" \
    --add-mamba "pytorch" \
    --run-tests

# Force a rebuild of the image layers
python -m modal_tf_builder.builder --force-rebuild --run-tests

# See all options
python -m modal_tf_builder.builder --help
```

### 2. Python Interface (Recommended for Modal Apps)
Import and use the `setup_modal_tf_gpu` function within your own Modal application scripts. Place your script either inside this repository directory or ensure the `modal-tf-gpu-builder` directory is in your Python path.

```python

# my_training_job.py
import modal

from modal_tf_builder import setup_modal_tf_gpu

# --- IMPORTANT ---
# Ensure the python_version here matches your LOCAL Python major.minor version
# if you plan to pass complex non-serializable objects.
# Sticking to the default '3.11' is generally safest if your local env is also 3.11.
builder_config = {
    "app_name": "my-training-run",
    "add_pip": ["wandb", "torchmetrics"], # Add project-specific dependencies
    "add_mamba": {"pytorch": None}, # Add other frameworks if needed
    "gpu_type": "T4" # Or "A10G", etc.
    # "run_tests": False
}

setup_data = setup_modal_tf_gpu(**builder_config)
app = setup_data["app"] # Get the configured Modal App object

# Define your Modal functions using the configured 'app'
@app.function(gpu=builder_config["gpu_type"]) # Might be good to match GPU type, "T4" by default
def train_model(hyperparams):
    import tensorflow as tf
    import pandas as pd
    import wandb
    print(f"Running training with TF {tf.__version__}")
    # ... your model training code ...
    # wandb.init(...)
    # model.fit(...)
    return {"status": "completed"}

@app.local_entrypoint()
def main(config_file: str = "params.yaml"):
    # Load hyperparameters or data paths
    params = {"lr": 0.001} # Example
    print(f"Starting remote training job for app: {app.name}")
    result = train_model.remote(params)
    print(f"Job finished with result: {result}")
```

## Configuration
- Default package versions (TF, CUDA, NumPy, etc.) are set within `modal_tf_builder/builder.py`. These defaults are based on a known stable combination.
- Use `--add-apt`, `--add-mamba`, and `--add-pip` (CLI) or the corresponding arguments in the `setup_modal_tf_gpu` function (Python) to add packages.
- For `--add-mamba` (CLI) or `add_mamba_packages` (Python):
- Use `package_name=version` to specify a version (e.g., `numpy=1.25.0`). This will override the default version if the package exists in the base list.
- Use `package_name` to install the latest compatible version according to the specified channels.

## Python Version Matching (Crucial!)
Modal serializes (pickles) Python objects to send data between your local machine and the remote container. This process can fail if the Python major.minor versions differ significantly (e.g., local 3.10 vs. remote 3.11).
- Recommendation: Ensure the Python version in your local environment (where you run `modal run ...`) matches the `python_version` argument passed to `setup_modal_tf_gpu` (default is "3.11"). Patch versions (e.g., 3.11.8 vs 3.11.9) are usually compatible.

## License

Distributed under the MIT License. See LICENSE for more information.