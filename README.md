# Modal TensorFlow GPU Environment Builder (Python 3.11)

**Author:** Osvald Nigola ([ozzuke](https://github.com/ozzuke))

**Version:** 0.2.0

## Problem Solved

Setting up a reliable environment for running TensorFlow with GPU acceleration on Modal.io can be tricky due to complex dependencies. This tool automates the creation of a **stable Python 3.11** Modal Image and App, specifically tailored for university assignments or projects needing a functional TF+GPU setup quickly.

## Features

*   Creates stable, reproducible **Python 3.11** TF+GPU environments on Modal using Micromamba.
*   Based on a known working combination (TF 2.14, CUDA 11.8, Python 3.11).
*   Easily add custom `apt`, `micromamba`, and `pip` packages to the base image.
*   Includes optional verification tests to confirm GPU functionality and provide timing context.
*   Provides both a Command Line Interface (CLI) for testing and a Python function for easy integration into your Modal scripts.
*   Designed for straightforward setup and usage via `pip install`.

## Default Environment

The base image created by this builder includes the following core components (installed via Micromamba):

*   **Python:** 3.11.x
*   **CUDA Toolkit:** 11.8.x (`cudatoolkit=11.8`)
*   **TensorFlow (GPU):** 2.14.0 (`tensorflow-gpu=2.14.0`)
*   **NumPy:** 1.26.4 (`numpy=1.26.4`)

It also includes the latest compatible versions (at build time) of these common libraries:
*   `cuda-nvcc` (CUDA compiler, version tied to cudatoolkit)
*   `cudnn`
*   `keras` (Usually managed by TensorFlow)
*   `scipy`
*   `pandas`
*   `pyarrow`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn`
*   `Pillow` (PIL fork for image processing)
*   `tqdm` (Progress bars)
*   `transformers` (Hugging Face)
*   `datasets` (Hugging Face)

System libraries installed via `apt-get`:
*   `libquadmath0`, `libgomp1`, `libgfortran5` (Required by NumPy/SciPy)

You can add more packages or override versions using the provided options (see Usage).

## Prerequisites (Strict)

*   **Python 3.11:** You **MUST** have Python 3.11.x installed locally and use it in your environment. This is critical because Modal requires matching Python major.minor versions between your local machine and the remote container to avoid errors when transferring data.
*   **pip:** Comes with Python 3.11.
*   **Git:** For cloning the repository or installing directly.
*   **Modal Account:** A configured Modal account (`modal token set ...`).

## Installation / Setup

Choose **one** of the following methods. Both require an active **Python 3.11** environment.

**Method 1: Direct Install from GitHub (Recommended for Users)**

This is the simplest way to use the builder in your projects.

1.  **Create & Activate Python 3.11 Environment:**
    *   Using `venv`:
        ```bash
        # Make sure python3.11 points to your Python 3.11 installation
        python3.11 -m venv my_ai_project_env
        source my_ai_project_env/bin/activate # On Windows use my_ai_project_env\Scripts\activate
        ```
    *   Using `conda`:
        ```bash
        conda create -n my_ai_project_env python=3.11 -y
        conda activate my_ai_project_env
        ```
2.  **Install Modal Runtime Dependencies:**
    ```bash
    pip install modal tblib
    ```
3.  **Install the Builder from GitHub:**
    ```bash
    pip install git+https://github.com/ozzuke/modal-tf-gpu-builder.git
    ```

**Method 2: Editable Install (Recommended for Development/Contribution)**

Use this if you want to modify the builder's code.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/ozzuke/modal-tf-gpu-builder.git
    cd modal-tf-gpu-builder
    ```
2.  **Create & Activate Python 3.11 Environment (inside the repo dir):**
    ```bash
    python3.11 -m venv .venv
    source .venv/bin/activate
    # Or use conda create -n modal-builder-dev python=3.11 && conda activate modal-builder-dev
    ```
3.  **Install in Editable Mode:**
    ```bash
    pip install -e .
    ```
4.  **Install Modal Runtime Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Activate your Python 3.11 environment** where you installed the builder.
2.  Import and use the `setup_modal_tf_gpu` function in your Modal scripts.

```python
# Example: my_tf_assignment.py
import modal
import sys # Added to show Python version in container

# Import the installed builder package
from modal_tf_gpu_builder import setup_modal_tf_gpu

# --- Configuration ---
# Builder uses Python 3.11 by default
GPU_TYPE = "T4" # Or "A10G", etc.

builder_config = {
    "app_name": "my-tf-assignment-app",
    "gpu_type": GPU_TYPE,
    # Add packages needed for YOUR assignment (beyond the defaults)
    "add_pip": ["wandb", "tensorflow_datasets"], # Example: Add experiment tracking and TFDS
    "add_mamba": {"scikit-image": None} # Example: Add scikit-image
}

# --- Setup Modal App ---
# This uses the builder to configure an app with a Python 3.11 TF+GPU image
print(f"Configuring Modal App: {builder_config['app_name']}")
setup_data = setup_modal_tf_gpu(**builder_config)
app = setup_data["app"] # Get the configured app object

# --- Define Your Modal Function(s) ---
@app.function(gpu=GPU_TYPE, timeout=600)
def run_assignment_task(data_url: str):
    import tensorflow as tf
    import pandas as pd # Already included by default
    import wandb # Added via add_pip
    import tensorflow_datasets as tfds # Added via add_pip
    from skimage import io # Added via add_mamba

    print(f"Running task with TF {tf.__version__} in Python {sys.version}")
    print(f"Wandb version: {wandb.__version__}")
    # ... download data using data_url ...
    # ... load data with pandas ...
    # ... process images with skimage ...
    # ... load TFDS dataset ...
    # ... build/train your TF model ...
    # ... log metrics with wandb ...
    # ... return results ...
    return {"status": "completed"}

# --- Local Entrypoint ---
@app.local_entrypoint()
def main(data_source: str = "http://example.com/my_data.csv"):
    print("Starting Modal task...")
    # Ensure you are running this local entrypoint using your Python 3.11 env!
    result = run_assignment_task.remote(data_source)
    print(f"Modal task finished: {result}")

```

### Command Line Interface (CLI)

You can also run the builder from the command line (primarily for testing the build process itself). Make sure your **Python 3.11 environment** (where the builder is installed) is active.

```bash
# Test the default Python 3.11 TF+GPU build and run verification
python -m modal_tf_builder.builder --run-tests --verbose-tests

# Test adding packages and run verification
python -m modal_tf_builder.builder --add-pip "requests" --add_mamba "opencv" --run-tests
```

## Configuration Options

When calling `setup_modal_tf_gpu` in Python or using the CLI:

*   `app_name` (str): Name for the Modal App (default: "tf-gpu-app").
*   `gpu_type` (str): GPU type for build and execution (e.g., "T4", "A10G", "H100", default: "T4").
*   `add_apt_packages` (List[str]): Additional apt packages to install (CLI: `--add-apt pkg1 pkg2`).
*   `add_mamba_packages` (Dict[str, Optional[str]]): Additional micromamba packages. Use `{"pkg": "version"}` or `{"pkg": None}` for latest compatible (CLI: `--add-mamba pkg1=1.2.3 pkg2`). These override defaults if the package name matches.
*   `add_pip_packages` (List[str]): Additional pip packages (CLI: `--add-pip pkg1 pkg2`).
*   `mamba_channels` (List[str]): Override default micromamba channels (CLI: `--mamba-channels channel1 channel2`).
*   `run_tests` (bool): Run GPU verification tests after setup (CLI: `--run-tests`).
*   `verbose_tests` (bool): Show verbose output during tests (CLI: `--verbose-tests`).
*   `force_rebuild` (bool): Force Modal to rebuild image layers, ignoring cache (CLI: `--force-rebuild`).

## License

Distributed under the MIT License. See `LICENSE` for more information.