# Modal TensorFlow GPU Environment Builder (Python 3.11)

**Author:** Osvald Nigola ([ozzuke](https://github.com/ozzuke))

**Version:** 0.2.2

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
2.  **Install the Builder (includes dependencies `modal` and `tblib`):**
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
3.  **Install in Editable Mode (includes dependencies):**
    ```bash
    pip install -e .
    ```

## Usage in Python Scripts

1.  **Activate your Python 3.11 environment** where you installed the builder.
2.  Import and use the `setup_modal_tf_gpu` function in your Modal scripts.
3. Run the script with Modal using `modal run my_tf_assignment.py`.

```python
# Example: my_tf_assignment.py
import modal
import sys

# Import the installed builder package
from modal_tf_builder import setup_modal_tf_gpu

# --- Configuration ---
# Builder uses Python 3.11 by default
GPU_TYPE = "T4" # Or "A10G", etc.

builder_config = {
    "app_name": "my-tf-assignment-app",
    "gpu_type": GPU_TYPE,
    # Add packages needed for YOUR assignment (beyond the defaults)
    "add_pip_packages": ["wandb", "tensorflow_datasets"], # Example: Add experiment tracking and TFDS
    "add_mamba_packages": {"scikit-image": None} # Example: Add scikit-image
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

## Usage via Command Line (CLI)

The CLI is primarily for **testing the build process** for a given configuration. It builds the image (if not cached) and optionally runs the verification tests.

Make sure your **Python 3.11 environment** (where the builder is installed) is active before running these commands.

**Basic Test:** Build the default image and run verification tests with detailed output.
```bash
python -m modal_tf_builder.builder --run-tests --verbose-tests
```

**Test Adding Packages:** Build an image with additional packages (`requests` via pip, `opencv` via mamba) and run tests.
```bash
python -m modal_tf_builder.builder --add-pip requests --add-mamba opencv --run-tests
```

**Test with a Specific GPU:** Build for an A10G GPU and run tests.
```bash
python -m modal_tf_builder.builder --gpu-type A10G --run-tests
```

**Force Rebuild:** Ignore the cache and rebuild the image layers from scratch (useful for debugging build issues).
```bash
python -m modal_tf_builder.builder --force-rebuild --run-tests
```

**Build Only (No Tests):** Just trigger the image build process without running verification.
```bash
python -m modal_tf_builder.builder
```

## Usage in Jupyter / IPython

You can use the builder interactively within a Jupyter Notebook or IPython session.

**Important:**
*   Start Jupyter Lab/Notebook or IPython from your activated **Python 3.11 environment**.
*   Modal functions (`@app.function`) must generally be defined in the global scope of a cell.
*   **Crucially, remote function calls (`.remote()`) must be wrapped in `with app.run():`** to execute within an active Modal app context in interactive sessions. Use `with modal.enable_output():` as well to see logs.

```python
# Cell 1: Imports and Setup
import modal
import sys
import time
# Import the installed builder package
from modal_tf_builder import setup_modal_tf_gpu

print("Configuring Modal app for interactive use...")
# Configure the builder
interactive_config = {
    "app_name": "interactive-tf-session",
    "gpu_type": "T4",
    "add_pip_packages": ["ipywidgets"], # Add notebook specific things if needed
}

# Call the builder - this defines the image and app object
setup_data = setup_modal_tf_gpu(**interactive_config)
app = setup_data["app"]

print(f"Modal App '{app.name}' configured with Python 3.11 TF+GPU image.")

# Cell 2: Define a Modal Function
# This function will run remotely on Modal using the image defined above
@app.function(gpu=interactive_config["gpu_type"], timeout=300)
def check_tf_version_remote():
    import tensorflow as tf
    import platform
    import time
    # Simulate some work
    print("Remote function started...")
    time.sleep(5)
    print("Remote function finishing.")
    return {
        "tf_version": tf.__version__,
        "python_version": platform.python_version(),
        "cuda_build_info": tf.sysconfig.get_build_info()
    }

# Cell 3: Run the Modal Function Remotely
print("Calling remote function within app.run()...")
# Use .remote() INSIDE the app.run() context manager
# Use modal.enable_output() to see logs from the remote function
result = None # Define outside the block
with modal.enable_output():
    with app.run():
        result = check_tf_version_remote.remote()
print("Remote function finished.")

# Cell 4: Display Results
if result:
    print("\nResults from Modal:")
    print(f" TensorFlow Version: {result.get('tf_version', 'N/A')}")
    print(f" Python Version (in container): {result.get('python_version', 'N/A')}")
    print(f" CUDA Version (TF build): {result.get('cuda_build_info', {}).get('cuda_version', 'N/A')}")
    print(f" cuDNN Version (TF build): {result.get('cuda_build_info', {}).get('cudnn_version', 'N/A')}")
else:
    print("Failed to get results.")


# Cell 5: Define another function (e.g., simple GPU task)
@app.function(gpu=interactive_config["gpu_type"])
def simple_gpu_task(size=1000):
    import tensorflow as tf
    import time
    print(f"Starting GPU task with size {size}x{size}...")
    start = time.time()
    with tf.device('/GPU:0'):
        a = tf.random.normal((size, size))
        b = tf.random.normal((size, size))
        c = tf.matmul(a, b).numpy() # Force execution
    end = time.time()
    print("GPU task complete.")
    return {"time_taken": end - start, "output_shape": c.shape}

# Cell 6: Run the GPU task (Corrected)
print("Running simple GPU task within app.run()...")
gpu_result = None # Define outside the block
with modal.enable_output():
    with app.run():
        gpu_result = simple_gpu_task.remote(size=2000)

if gpu_result:
    print(f"GPU task finished in {gpu_result.get('time_taken', -1):.4f}s")
else:
    print("Failed to get GPU task results.")

```

## Configuration Options

When calling `setup_modal_tf_gpu` in Python or using the CLI:

*   `app_name` (str): Name for the Modal App (default: "tf-gpu-app").
*   `gpu_type` (str): GPU type for build and execution (e.g., "T4", "A10G", "H100", default: "T4").
*   `add_apt_packages` (List[str]): Additional apt packages to install (CLI: `--add-apt pkg1 pkg2`).
*   `add_mamba_packages` (Dict[str, Optional[str]]): Additional micromamba packages. Use `{"pkg": "version"}` or `{"pkg": None}` for latest compatible (CLI: `--add-mamba pkg1=1.2.3 pkg2`). These override defaults if the package name matches.
*   `add_pip_packages` (List[str]): Additional pip packages (CLI: `--add-pip pkg1 pkg2`).
*   `mamba_channels` (List[str]): Override default micromamba channels (CLI: `--mamba-channels channel1 channel2`).
*   `run_tests` (bool): Run GPU verification tests after setup (CLI: `--run-tests`). Only applicable when run via CLI.
*   `verbose_tests` (bool): Show verbose output during tests (CLI: `--verbose-tests`). Only applicable when run via CLI.
*   `force_rebuild` (bool): Force Modal to rebuild image layers, ignoring cache (CLI: `--force-rebuild`).

## License

Distributed under the MIT License. See `LICENSE` for more information.