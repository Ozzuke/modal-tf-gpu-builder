# Author: Osvald Nigola
# Core logic for building the Modal TF GPU environment.

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

# Default versions (based on known working setup)
DEFAULT_PYTHON_VERSION: str = "3.11"
DEFAULT_TF_VERSION: str = "2.14.0"
DEFAULT_NUMPY_VERSION: str = "1.26.4"
DEFAULT_CUDA_VERSION: str = "11.8"
DEFAULT_CUDNN_VERSION: str = "8.6"

# Default base packages
DEFAULT_APT_PACKAGES: List[str] = ["libquadmath0", "libgomp1", "libgfortran5"] # Required by NumPy
DEFAULT_MAMBA_PACKAGES: Dict[str, Optional[str]] = {
    # Core ML & GPU
    "cudatoolkit": DEFAULT_CUDA_VERSION,
    "cudnn": DEFAULT_CUDNN_VERSION,
    "cuda-nvcc": None,
    "tensorflow-gpu": DEFAULT_TF_VERSION,
    "keras": None,

    # Data Handling & Scientific Computing
    "numpy": DEFAULT_NUMPY_VERSION,
    "scipy": None,
    "pandas": None,
    "pyarrow": None,

    # Plotting & Visualization
    "matplotlib": None,
    "seaborn": None,

    # ML Utilities
    "scikit-learn": None,
    "Pillow": None,
    "tqdm": None,

    # Hugging Face Ecosystem
    "transformers": None,
    "datasets": None,
}
DEFAULT_PIP_PACKAGES: List[str] = []
DEFAULT_MAMBA_CHANNELS: List[str] = ["conda-forge", "nvidia", "defaults"]

# --- Helper Functions ---

def check_local_environment(required_py_version_prefix: str) -> bool:
    """
    Checks the local Python environment for compatibility and necessary tools.

    Args:
        required_py_version_prefix: The required major.minor Python version (e.g., "3.11").

    Returns:
        True if critical checks pass, False otherwise.
    """
    print("\n--- Checking Local Environment ---")
    passed = True

    # 1. Python Version Check
    local_py_version = platform.python_version()
    local_py_prefix = ".".join(local_py_version.split('.')[:2])
    print(f"Local Python Version: {local_py_version}")
    if local_py_prefix != required_py_version_prefix:
        print(f"!! WARNING !!: Local Python ({local_py_prefix}) doesn't match target image Python ({required_py_version_prefix}).")
        print("  This can cause Modal serialization errors if passing complex objects.")
        print("  It's highly recommended to align these versions.")
        # passed = False # Decide if this should be a hard failure

    # 2. Modal Client Check
    try:
        modal_version = importlib.metadata.version("modal-client")
        print(f"Modal Client Version: {modal_version}")
    except importlib.metadata.PackageNotFoundError:
        print("ERROR: modal-client not found locally. Please install it (`pip install modal-client`).")
        passed = False

    # 3. tblib Check (Recommended for Modal stability)
    try:
        importlib.metadata.version("tblib")
        print("tblib: Found (Recommended)")
    except importlib.metadata.PackageNotFoundError:
        print("WARNING: tblib not found locally. Install with 'pip install tblib' to potentially improve Modal stability.")

    # 4. Conda Check (Informational)
    try:
        # Use shell=True for broader compatibility, though less secure if path isn't controlled
        conda_version = subprocess.check_output("conda --version", shell=True, text=True, stderr=subprocess.DEVNULL).strip()
        print(f"Conda Version: {conda_version}")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("INFO: Conda command not found or failed. Not critical if not using local conda envs.")

    print("--- Local Environment Check Complete ---")
    return passed

# --- Main Setup Function ---

def setup_modal_tf_gpu(
    app_name: str = "tf-gpu-app",
    python_version: str = DEFAULT_PYTHON_VERSION,
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
    Builds a configurable Modal Image and App for TensorFlow GPU tasks.

    Args:
        app_name: Name for the Modal App.
        python_version: Python version for the container (e.g., "3.11").
        base_apt_packages: Default list of apt packages.
        add_apt_packages: List of additional apt packages to install.
        base_mamba_packages: Default dict of micromamba packages {pkg: version or None}.
        add_mamba_packages: Dict of additional micromamba packages, potentially overriding base versions.
        base_pip_packages: Default list of pip packages.
        add_pip_packages: List of additional pip packages to install.
        mamba_channels: List of micromamba channels to use.
        gpu_type: Type of GPU to request for builds and functions (e.g., "T4", "A10G").
        run_tests: If True, run verification tests after building the image.
        verbose_tests: If True and run_tests is True, show detailed test output.
        force_rebuild: If True, force Modal to rebuild the image layers.

    Returns:
        A dictionary containing:
            'image': The configured modal.Image object.
            'app': The configured modal.App object.
            'test_results': Results from the verification tests (if run), or None.
    """
    print(f"\n--- Configuring Modal App: {app_name} ---")
    print(f"Target Python Version: {python_version}")
    print(f"Target GPU Type: {gpu_type}")

    # 1. Combine Package Lists
    final_apt = sorted(list(set(base_apt_packages + add_apt_packages))) # Use set to remove duplicates, sort for consistency

    final_mamba = base_mamba_packages.copy()
    final_mamba.update(add_mamba_packages) # add_mamba overrides base

    final_pip = sorted(list(set(base_pip_packages + add_pip_packages)))

    print(f"Final APT Packages: {final_apt or 'None'}")
    print(f"Final Micromamba Packages: {final_mamba or 'None'}")
    print(f"Final PIP Packages: {final_pip or 'None'}")
    print(f"Micromamba Channels: {mamba_channels}")

    # 2. Build the Image Definition
    image = modal.Image.micromamba(python_version=python_version)

    if final_apt:
        print("Adding APT packages...")
        image = image.apt_install(*final_apt, force_build=force_rebuild)

    if final_mamba:
        print("Adding Micromamba packages...")
        mamba_install_args = []
        # Sort items for deterministic image layer hashing (helps caching)
        for pkg, version in sorted(final_mamba.items()):
            if version:
                mamba_install_args.append(f"{pkg}={version}")
            else:
                mamba_install_args.append(pkg)
        image = image.micromamba_install(
            *mamba_install_args,
            channels=mamba_channels,
            gpu=gpu_type, # Specify GPU for build step if needed
            force_build=force_rebuild,
        )

    if final_pip:
        print("Adding PIP packages...")
        image = image.pip_install(*final_pip, force_build=force_rebuild)

    # Add environment variables if needed - could be made configurable too
    # Example: Pinning CUDA paths if auto-detection fails
    # cuda_path = f"/opt/conda/pkgs/cuda-nvcc-{DEFAULT_CUDA_VERSION}.0-0/" # Adjust if needed
    # image = image.env({
    #     "XLA_FLAGS": f"--xla_gpu_cuda_data_dir={cuda_path}",
    #     "LD_LIBRARY_PATH": "$LD_LIBRARY_PATH:/opt/conda/lib/",
    #     "CUDA_HOME": cuda_path
    # })

    print("Image definition created.")

    # 3. Create the App
    # Use the image object directly when creating the app
    app = modal.App(name=app_name, image=image)
    print(f"Modal App '{app.name}' created.")

    # 4. Define and Optionally Run Tests (Nested function to access 'app')
    @app.function(gpu=gpu_type, timeout=300)
    def _run_gpu_verification_tests(verbose: bool = False):
        """Internal function to verify GPU setup within the Modal container."""
        results = {"success": False, "details": {}, "logs": []}
        def log_print(msg):
            print(msg) # Print within the container log
            results["logs"].append(msg)

        try:
            log_print("Importing libraries...")
            import tensorflow as tf
            import numpy as np
            import os
            import time # Ensure time is imported
            log_print("Imports successful.")

            # Version Info
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

            # GPU Check
            gpu_devices = tf.config.list_physical_devices('GPU')
            results['details']['gpu_devices'] = [str(d) for d in gpu_devices]
            log_print(f"GPU Devices Found: {gpu_devices}")
            if not gpu_devices:
                log_print("ERROR: No GPU devices found by TensorFlow.")
                results['details']['error'] = "No GPU detected by TensorFlow"
                return results # Early exit if no GPU

            # Basic Operation Test
            log_print("\nRunning basic matrix multiplication...")
            with tf.device('/GPU:0'):
                a = tf.random.normal((500, 500)) # Smaller for faster test
                b = tf.random.normal((500, 500))
                start_time = time.time()
                c = tf.matmul(a, b)
                _ = c.numpy() # Force execution
                op_time = time.time() - start_time
            results['details']['matmul_time_s'] = op_time
            log_print(f"Matrix multiplication successful (Time: {op_time:.4f}s)")

            # Keras Test
            log_print("\nRunning Keras model test...")
            model = tf.keras.Sequential([
                tf.keras.Input(shape=(5,)), # Use Input layer for clarity
                tf.keras.layers.Dense(10, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            model.compile(optimizer='sgd', loss='mse')
            x = np.random.random((50, 5)) # Smaller data
            y = np.random.random((50, 1))
            history = model.fit(x, y, epochs=2, verbose=1 if verbose else 0) # Control verbosity
            results['details']['keras_final_loss'] = history.history['loss'][-1]
            log_print("Keras training successful.")

            results['success'] = True

        except Exception as e:
            log_print(f"\nERROR during GPU verification: {e}")
            import traceback
            results['details']['error'] = str(e)
            results['details']['traceback'] = traceback.format_exc()
            if verbose:
                 log_print(results['details']['traceback']) # Print traceback if verbose

        return results

    test_results_data = None
    if run_tests:
        print("\n--- Running GPU Verification Tests ---")
        # Decide whether to show Modal's output based on verbosity
        # Use modal.enable_output() context manager for cleaner handling
        output_context = modal.enable_output if verbose_tests else lambda: open(os.devnull, 'w')
        try:
             with output_context():
                 # We need to run the app to invoke the function defined on it
                 with app.run(show_progress=verbose_tests):
                     test_results_data = _run_gpu_verification_tests.remote(verbose=verbose_tests)
        except Exception as e:
             print(f"ERROR: Failed to launch or run Modal tests: {e}")
             test_results_data = {"success": False, "details": {"error": f"Modal execution failed: {e}"}, "logs": []}

        if test_results_data:
            print(f"Test Success: {test_results_data.get('success', False)}")
            if not test_results_data.get('success') and not verbose_tests:
                 # Print logs only if failed and not already verbose
                 print("Test Logs (on failure):")
                 for log_line in test_results_data.get('logs', []):
                      print(f"  {log_line}")
        else:
            print("WARNING: Test execution did not return results.")
        print("--- Test Run Complete ---")


    return {
        "image": image, # Return the image definition object
        "app": app,     # Return the app object
        "test_results": test_results_data,
    }

# --- Command Line Interface ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Setup and optionally test a Modal TensorFlow GPU Environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core Config
    parser.add_argument("--app-name", default="tf-gpu-app", help="Name for the Modal App.")
    parser.add_argument("--python-version", default=DEFAULT_PYTHON_VERSION, help="Python version for the container.")
    parser.add_argument("--gpu-type", default="T4", help="GPU type for build and execution (e.g., T4, A10G, H100).")

    # Package Management
    parser.add_argument("--add-apt", nargs='*', default=[], help="Additional apt packages to install.")
    # For dicts via CLI, a simple approach is key=value pairs
    parser.add_argument("--add-mamba", nargs='*', default=[], help="Additional mamba packages (e.g., 'package_name=1.2.3' or 'package_name' for latest).")
    parser.add_argument("--add-pip", nargs='*', default=[], help="Additional pip packages to install.")
    parser.add_argument("--mamba-channels", nargs='*', default=DEFAULT_MAMBA_CHANNELS, help="Micromamba channels.")

    # Testing and Build Control
    parser.add_argument("--run-tests", action="store_true", help="Run GPU verification tests after setup.")
    parser.add_argument("--verbose-tests", action="store_true", help="Show verbose output during tests (implies Modal output).")
    parser.add_argument("--force-rebuild", action="store_true", help="Force Modal to rebuild image layers.")

    args = parser.parse_args()

    # --- Pre-checks ---
    py_version_prefix = ".".join(args.python_version.split('.')[:2])
    if not check_local_environment(py_version_prefix):
        # Warning is printed in the function, decide if exit is needed
        # print("ERROR: Local environment checks failed. Please address critical issues.")
        # sys.exit(1) # Make this exit optional? For now, just warn.
        pass


    # --- Parse Dict Args ---
    add_mamba_dict = {}
    for item in args.add_mamba:
        if '=' in item:
            pkg, version = item.split('=', 1)
            add_mamba_dict[pkg] = version
        else:
            add_mamba_dict[item] = None # Request latest compatible

    # --- Run Setup ---
    # Call the main setup function
    setup_info = setup_modal_tf_gpu(
        app_name=args.app_name,
        python_version=args.python_version,
        # base packages are taken from defaults in the function
        add_apt_packages=args.add_apt,
        add_mamba_packages=add_mamba_dict,
        add_pip_packages=args.add_pip,
        mamba_channels=args.mamba_channels,
        gpu_type=args.gpu_type,
        run_tests=args.run_tests,
        verbose_tests=args.verbose_tests,
        force_rebuild=args.force_rebuild,
    )

    # --- Output ---
    # The primary purpose of running the script directly is usually testing the build.
    # The 'app' and 'image' objects are created but not typically used further here.
    print("\n--- Setup Script Finished ---")
    print(f"Modal App Name configured as: {setup_info['app'].name}")
    if setup_info['test_results']:
        print(f"Test Run Status: {'Success' if setup_info['test_results'].get('success') else 'Failed or Incomplete'}")
    elif args.run_tests:
        print("Test Run Status: Incomplete (No results returned)")

    if not args.run_tests:
        print("\nRun with --run-tests to verify the environment after setup.")

    print("\nSee README.md for instructions on using this builder in your Modal scripts.")
