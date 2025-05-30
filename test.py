import torch

print("--- PyTorch & CUDA ---")
print(f"PyTorch Version: {torch.__version__}")

if torch.cuda.is_available():
    print(f"CUDA available: True")
    print(f"PyTorch built with CUDA version: {torch.version.cuda}")
else:
    print(f"CUDA available: False (This is a CPU-only PyTorch build)")

# --- Vision & Medical Imaging Libraries ---
print("\n--- Vision & Medical Imaging Libraries ---")
try:
    import torchvision
    print(f"Torchvision Version: {torchvision.__version__}")
except ImportError:
    print("Torchvision is not installed.")
except Exception as e:
    print(f"Could not determine Torchvision version: {e}")

try:
    import monai
    print(f"MONAI Version: {monai.__version__}")
except ImportError:
    print("MONAI is not installed.")
except Exception as e:
    print(f"Could not determine MONAI version: {e}")

try:
    import PIL
    print(f"Pillow (PIL) Version: {PIL.__version__}")
except ImportError:
    print("Pillow (PIL) is not installed.")
except Exception as e:
    print(f"Could not determine Pillow (PIL) version: {e}")

try:
    import skimage
    print(f"scikit-image Version: {skimage.__version__}")
except ImportError:
    print("scikit-image is not installed.")
except Exception as e:
    print(f"Could not determine scikit-image version: {e}")


# --- NLP & CLIP Libraries ---
print("\n--- NLP & CLIP Libraries ---")
try:
    import transformers
    print(f"Transformers Version: {transformers.__version__}")
except ImportError:
    print("Transformers is not installed.")
except Exception as e:
    print(f"Could not determine Transformers version: {e}")

try:
    import open_clip
    print(f"OpenCLIP Version: {open_clip.__version__}")
except ImportError:
    print("OpenCLIP is not installed.")
except AttributeError: # open_clip might not have __version__ directly, try getting from package if installed
    try:
        import pkg_resources
        print(f"OpenCLIP Version: {pkg_resources.get_distribution('open_clip_torch').version}")
    except Exception:
        print("OpenCLIP is installed, but version attribute not found and pkg_resources could not determine it.")
except Exception as e:
    print(f"Could not determine OpenCLIP version: {e}")

# --- Data Handling & Utility Libraries ---
print("\n--- Data Handling & Utility Libraries ---")
try:
    import pandas as pd
    print(f"Pandas Version: {pd.__version__}")
except ImportError:
    print("Pandas is not installed.")
except Exception as e:
    print(f"Could not determine Pandas version: {e}")

try:
    import numpy as np
    print(f"NumPy Version: {np.__version__}")
except ImportError:
    print("NumPy is not installed.")
except Exception as e:
    print(f"Could not determine NumPy version: {e}")

try:
    import sklearn
    print(f"scikit-learn Version: {sklearn.__version__}")
except ImportError:
    print("scikit-learn is not installed.")
except Exception as e:
    print(f"Could not determine scikit-learn version: {e}")

try:
    import matplotlib
    print(f"Matplotlib Version: {matplotlib.__version__}")
except ImportError:
    print("Matplotlib is not installed.")
except Exception as e:
    print(f"Could not determine Matplotlib version: {e}")

try:
    import tqdm
    print(f"tqdm Version: {tqdm.__version__}")
except ImportError:
    print("tqdm is not installed.")
except Exception as e:
    print(f"Could not determine tqdm version: {e}")

print("\n--- End of Library Version Check ---")