# Author: Osvald Nigola
# Core logic for building the Modal TF GPU environment (Fixed to Python 3.11).

import modal
import argparse
import sys
import platform
import subprocess
import importlib.metadata
import time
import os
from typing import List, Dict, Optional, Any

# --- Configuration ---

# Fixed Python version for stability and ease of use
DEFAULT_PYTHON_VERSION: str = "3.11"
# Default package versions based on a known working TF 2.14 setup
DEFAULT_TF_VERSION: str = "2.14.0"
DEFAULT_NUMPY_VERSION: str = "1.26.4" # Compatible with TF 2.14
DEFAULT_CUDA_VERSION: str = "11.8" # Required by TF 2.14
# DEFAULT_CUDNN_VERSION: str = "8.6" # Compatible with CUDA 11.8
DEFAULT_CUDNN_VERSION: str = None # Errors finding provider for 8.6

# Default system packages (required by NumPy/SciPy)
DEFAULT_APT_PACKAGES: List[str] = ["libquadmath0", "libgomp1", "libgfortran5"]

# Default core Python packages installed via Micromamba
DEFAULT_MAMBA_PACKAGES: Dict[str, Optional[str]] = {
    # GPU Stack
    "cudatoolkit": DEFAULT_CUDA_VERSION,
    "cudnn": DEFAULT_CUDNN_VERSION,
    "cuda-nvcc": None, # Version usually tied to cudatoolkit

    # Core ML
    "tensorflow-gpu": DEFAULT_TF_VERSION,
    "keras": None, # Let TF handle its Keras version

    # Data Science & Numerics
    "numpy": DEFAULT_NUMPY_VERSION,
    "scipy": None,
    "pandas": None,
    "pyarrow": None, # Efficient data serialization (e.g., Parquet)
    "scikit-learn": None,

    # Plotting
    "matplotlib": None,
    "seaborn": None,

    # Utilities
    "Pillow": None, # Image handling
    "tqdm": None, # Progress bars

    # Hugging Face Ecosystem (Commonly used)
    "transformers": None,
    "datasets": None,
}

# Default pip packages (can be added to if needed)
DEFAULT_PIP_PACKAGES: List[str] = []

# Default Micromamba channels
DEFAULT_MAMBA_CHANNELS: List[str] = ["conda-forge", "nvidia", "defaults"]


# --- Helper Functions ---

def check_local_environment() -> bool:
    """
    Checks the local Python environment for Python version and necessary tools.

    Returns:
        True if critical checks pass, False otherwise.
    """
    print("\n--- Checking Local Environment ---")
    passed = True
    target_py_prefix = "3.11" # This tool requires Python 3.11 locally

    # 1. Python Version Check (Strict)
    local_py_version = platform.python_version()
    local_py_prefix = ".".join(local_py_version.split('.')[:2])
    print(f"Local Python Version: {local_py_version}")
    if local_py_prefix != target_py_prefix:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!! CRITICAL ERROR: Local Python is {local_py_prefix}, but this tool REQUIRES Python {target_py_prefix}. !!")
        print(f"!! Using mismatched versions WILL cause Modal errors. Please create and use !!")
        print(f"!! a Python {target_py_prefix} environment (venv or conda) for this project.          !!")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        passed = False # Mark check as failed

    # 2. Modal Package Check
    try:
        modal_version = importlib.metadata.version("modal")
        print(f"Modal Version: {modal_version}")
    except importlib.metadata.PackageNotFoundError:
        print("ERROR: 'modal' package not found locally. Please install it (`pip install modal`).")
        passed = False

    # 3. tblib Check (Recommended)
    try:
        importlib.metadata.version("tblib")
        print("tblib: Found (Recommended)")
    except importlib.metadata.PackageNotFoundError:
        print("WARNING: tblib not found locally. Install with 'pip install tblib'.") # Non-critical warning

    # 4. Conda Check (Informational)
    try:
        conda_version = subprocess.check_output("conda --version", shell=True, text=True, stderr=subprocess.DEVNULL).strip()
        print(f"Conda Version: {conda_version}")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("INFO: Conda command not found or failed (not required).")

    print("--- Local Environment Check Complete ---")
    return passed

# --- Global Test Function Definition ---
# This function must be defined globally for Modal to serialize it correctly
# when associated with an app instance later.
def _run_gpu_verification_tests_global(verbose: bool = False):
    """
    Internal function (global scope) to verify GPU setup within the Modal container.
    Runs basic TF/GPU checks and a small Keras model training.
    """
    results = {"success": False, "details": {}, "logs": []}
    def log_print(msg):
        """Helper to print logs inside the container and capture them."""
        print(msg)
        results["logs"].append(msg)

    try:
        log_print("Importing libraries...")
        import tensorflow as tf
        import numpy as np
        import os
        import time
        log_print("Imports successful.")

        # Version Info Check
        tf_version = tf.__version__
        np_version = np.__version__
        cuda_info = tf.sysconfig.get_build_info()
        cuda_version = cuda_info.get('cuda_version', 'N/A')
        cudnn_version = cuda_info.get('cudnn_version', 'N/A')
        results['details']['versions'] = {
            'tensorflow': tf_version, 'numpy': np_version,
            'cuda': cuda_version, 'cudnn': cudnn_version
        }
        log_print(f"Versions: TF={tf_version}, NumPy={np_version}, CUDA={cuda_version}, cuDNN={cudnn_version}")

        # GPU Device Check
        gpu_devices = tf.config.list_physical_devices('GPU')
        results['details']['gpu_devices'] = [str(d) for d in gpu_devices]
        log_print(f"GPU Devices Found: {gpu_devices}")
        if not gpu_devices:
            log_print("ERROR: No GPU devices found by TensorFlow.")
            results['details']['error'] = "No GPU detected by TensorFlow"
            return results # Test fails if no GPU

        # Basic GPU Operation Test (Matrix Multiplication)
        log_print("\nRunning basic matrix multiplication on GPU...")
        try:
            with tf.device('/GPU:0'):
                a = tf.random.normal((500, 500), dtype=tf.float32)
                b = tf.random.normal((500, 500), dtype=tf.float32)
                start_time = time.time()
                c = tf.matmul(a, b)
                # Force execution and sync by converting to numpy
                _ = c.numpy()
                op_time = time.time() - start_time
            results['details']['matmul_time_s'] = op_time
            log_print(f"Matrix multiplication successful (Time: {op_time:.4f}s)")

            # --- CONTEXT ---
            log_print(f"  -> INFO: Typical times for this test vary by GPU:")
            log_print(f"     - High-end (A100, H100): Usually << 0.1s")
            log_print(f"     - Mid-range (T4, A10G): ~0.05s - 0.2s")
            log_print(f"     - Older/Lower-end: May be longer.")
            log_print(f"     If the time seems excessive for the requested GPU type, it might indicate an issue.")
            # --- END CONTEXT ---

        except Exception as matmul_error:
            log_print(f"ERROR during matrix multiplication test: {matmul_error}")
            results['details']['error'] = f"Matmul test failed: {matmul_error}"
            return results # Fail test early


        # Simple Keras Model Training Test
        log_print("\nRunning Keras model test...")
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(5,)),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='sgd', loss='mse')
        # Use small random data for quick test
        x = np.random.random((50, 5))
        y = np.random.random((50, 1))
        history = model.fit(x, y, epochs=2, verbose=1 if verbose else 0) # Control Keras verbosity
        results['details']['keras_final_loss'] = history.history['loss'][-1]
        log_print("Keras training successful.")

        results['success'] = True # Mark test as successful if all steps passed

    except Exception as e:
        log_print(f"\nERROR during GPU verification: {e}")
        import traceback
        results['details']['error'] = str(e)
        results['details']['traceback'] = traceback.format_exc()
        if verbose:
             log_print(results['details']['traceback']) # Show full traceback if verbose

    return results


# --- Main Setup Function ---

def setup_modal_tf_gpu(
    app_name: str = "tf-gpu-app",
    base_apt_packages: List[str] = DEFAULT_APT_PACKAGES,
    add_apt_packages: List[str] = [],
    base_mamba_packages: Dict[str, Optional[str]] = DEFAULT_MAMBA_PACKAGES,
    add_mamba_packages: Dict[str, Optional[str]] = {},
    base_pip_packages: List[str] = DEFAULT_PIP_PACKAGES,
    add_pip_packages: List[str] = [],
    mamba_channels: List[str] = DEFAULT_MAMBA_CHANNELS,
    gpu_type: str = "T4",
    run_tests: bool = False,
    verbose_tests: bool = False,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    """
    Builds a configurable Modal Image (fixed to Python 3.11) and App
    for TensorFlow GPU tasks using Micromamba.

    Args:
        app_name: Name for the Modal App.
        base_apt_packages: Base list of apt packages.
        add_apt_packages: List of additional apt packages to install.
        base_mamba_packages: Base dict of micromamba packages {pkg: version or None}.
        add_mamba_packages: Dict of additional micromamba packages, potentially overriding base versions.
        base_pip_packages: Base list of pip packages.
        add_pip_packages: List of additional pip packages to install.
        mamba_channels: List of micromamba channels to use.
        gpu_type: Type of GPU to request for builds and functions (e.g., "T4", "A10G").
        run_tests: If True, run verification tests after building the image.
        verbose_tests: If True and run_tests is True, show detailed test output.
        force_rebuild: If True, force Modal to rebuild image layers.

    Returns:
        A dictionary containing:
            'image': The configured modal.Image object.
            'app': The configured modal.App object.
            'test_results': Results from the verification tests (if run), or None.
    """
    print(f"\n--- Configuring Modal App: {app_name} ---")
    print(f"Target Python Version: {DEFAULT_PYTHON_VERSION} (Fixed)")
    print(f"Target GPU Type: {gpu_type}")

    # 1. Combine and finalize package lists
    final_apt = sorted(list(set(base_apt_packages + add_apt_packages)))
    final_mamba = base_mamba_packages.copy()
    final_mamba.update(add_mamba_packages) # Add/override with user-provided mamba packages
    final_pip = sorted(list(set(base_pip_packages + add_pip_packages)))

    print(f"Final APT Packages: {final_apt or 'None'}")
    print(f"Final Micromamba Packages: {final_mamba or 'None'}")
    print(f"Final PIP Packages: {final_pip or 'None'}")
    print(f"Micromamba Channels: {mamba_channels}")

    # 2. Build the Modal Image definition using Micromamba
    image = modal.Image.micromamba(python_version=DEFAULT_PYTHON_VERSION) # Use fixed Python version

    if final_apt:
        print("Adding APT packages...")
        image = image.apt_install(*final_apt, force_build=force_rebuild)

    if final_mamba:
        print("Adding Micromamba packages...")
        mamba_install_args = []
        # Sort packages alphabetically for deterministic image layer hashing
        for pkg, version in sorted(final_mamba.items()):
            if version:
                mamba_install_args.append(f"{pkg}={version}")
            else:
                mamba_install_args.append(pkg) # Install latest compatible if version is None
        image = image.micromamba_install(
            *mamba_install_args,
            channels=mamba_channels,
            gpu=gpu_type, # Specify GPU for build step if needed (e.g., for CUDA libs)
            force_build=force_rebuild,
        )

    if final_pip:
        print("Adding PIP packages...")
        image = image.pip_install(*final_pip, force_build=force_rebuild)

    print("Adding modal_tf_builder package source to image...")
    image = image.add_local_python_source("modal_tf_builder")

    print("Image definition created.")

    # 3. Create the Modal App instance using the defined image
    app = modal.App(name=app_name, image=image)
    print(f"Modal App '{app.name}' created.")

    # 4. Associate the global test function with this specific app instance and its settings
    gpu_verification_runner = app.function(
        gpu=gpu_type,
        timeout=300 # Set a timeout for the test function
    )(_run_gpu_verification_tests_global)

    # 5. Optionally run the verification tests within a temporary app run
    test_results_data = None
    if run_tests:
        print("\n--- Running GPU Verification Tests ---")
        # Control Modal's output verbosity using modal.enable_output() context manager
        output_context = modal.enable_output if verbose_tests else lambda: open(os.devnull, 'w')
        try:
             with output_context():
                 # Run the app context manager to execute the remote function
                 with app.run():
                     test_results_data = gpu_verification_runner.remote(verbose=verbose_tests)
        except Exception as e:
             # Catch potential errors during Modal app launch or function execution
             print(f"ERROR: Failed to launch or run Modal tests: {e}")
             test_results_data = {"success": False, "details": {"error": f"Modal execution failed: {e}"}, "logs": []}

        # Report test results
        if test_results_data:
            print(f"Test Success: {test_results_data.get('success', False)}")
            # If tests failed and weren't verbose, print logs for debugging
            if not test_results_data.get('success') and not verbose_tests:
                 print("Test Logs (on failure):")
                 for log_line in test_results_data.get('logs', []):
                      print(f"  {log_line}")
        else:
            # Should not happen if try/except works, but good to check
            print("WARNING: Test execution did not return results.")
        print("--- Test Run Complete ---")

    # Return the configured image, app, and test results
    return {
        "image": image,
        "app": app,
        "test_results": test_results_data,
    }

# --- Command Line Interface ---
# Allows running the builder directly, primarily for testing the build process.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Setup and optionally test a Modal TensorFlow GPU Environment (Python 3.11).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core Configuration Arguments
    parser.add_argument("--app-name", default="tf-gpu-app", help="Name for the Modal App.")
    parser.add_argument("--gpu-type", default="T4", help="GPU type for build and execution (e.g., T4, A10G, H100).")

    # Package Management Arguments
    parser.add_argument("--add-apt", nargs='*', default=[], help="Additional apt packages to install.")
    parser.add_argument("--add-mamba", nargs='*', default=[], help="Additional mamba packages (e.g., 'pkg=1.2.3' or 'pkg').")
    parser.add_argument("--add-pip", nargs='*', default=[], help="Additional pip packages to install.")
    parser.add_argument("--mamba-channels", nargs='*', default=DEFAULT_MAMBA_CHANNELS, help="Micromamba channels to use.")

    # Testing and Build Control Arguments
    parser.add_argument("--run-tests", action="store_true", help="Run GPU verification tests after setup.")
    parser.add_argument("--verbose-tests", action="store_true", help="Show verbose output during tests (implies Modal output).")
    parser.add_argument("--force-rebuild", action="store_true", help="Force Modal to rebuild image layers, ignoring cache.")

    args = parser.parse_args()

    # --- Pre-execution Checks ---
    # Enforce local Python 3.11 environment when run via CLI
    if not check_local_environment():
        print("\nExiting due to local environment check failure (Python 3.11 required).")
        sys.exit(1)

    # --- Parse Mamba Dictionary Argument ---
    add_mamba_dict = {}
    for item in args.add_mamba:
        if '=' in item:
            pkg, version = item.split('=', 1)
            add_mamba_dict[pkg] = version
        else:
            add_mamba_dict[item] = None # Request latest compatible version

    # --- Execute Main Setup Function ---
    setup_info = setup_modal_tf_gpu(
        app_name=args.app_name,
        add_apt_packages=args.add_apt,
        add_mamba_packages=add_mamba_dict,
        add_pip_packages=args.add_pip,
        mamba_channels=args.mamba_channels,
        gpu_type=args.gpu_type,
        run_tests=args.run_tests,
        verbose_tests=args.verbose_tests,
        force_rebuild=args.force_rebuild,
    )

    # --- Final Output ---
    print("\n--- Setup Script Finished ---")
    print(f"Modal App Name configured as: {setup_info['app'].name}")
    if setup_info['test_results']:
        print(f"Test Run Status: {'Success' if setup_info['test_results'].get('success') else 'Failed or Incomplete'}")
    elif args.run_tests:
        # If tests were requested but no results dictionary was returned (e.g., early error)
        print("Test Run Status: Incomplete (No results returned)")

    if not args.run_tests:
        print("\nRun with --run-tests to build the image and verify the environment.")

    print("\nSee README.md for instructions on using this builder package in your Modal scripts.")