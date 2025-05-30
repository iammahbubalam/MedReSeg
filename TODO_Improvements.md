# Missing Features & Potential Improvements

This document outlines missing features and potential areas for improvement in the MedReSeg project.

## I. Checkpoint Management (`util/checkpoint_manager.py`)

-   **Robust Error Handling:**
    -   [ ] Handle potentially corrupted checkpoint files during loading.
    -   [ ] Add more specific error messages and recovery mechanisms.
-   **Structured Logging:**
    -   [ ] Replace `print` statements with a proper logging library (e.g., Python's `logging` module) for better control over log levels and output destinations.
-   **Flexible Checkpoint Naming/Retrieval:**
    -   [ ] Modify `get_latest_checkpoint` to find the most recent checkpoint based on timestamp or a more flexible naming pattern (e.g., `checkpoint_epoch_{epoch_number}_date_{date}.pth.tar`) rather than a fixed filename.
    -   [ ] Allow listing and choosing from multiple available checkpoints.
-   **Checkpoint Retention Strategy:**
    -   [ ] Implement a strategy to keep the last N checkpoints.
    -   [ ] Add functionality to save checkpoints every K epochs, in addition to the best one.
-   **Enhanced Metadata in Checkpoints:**
    -   [ ] Store more comprehensive metadata within the `state` dictionary, such as:
        -   Training arguments/hyperparameters.
        -   Model configuration details.
        -   Dataset information (e.g., name, version).
        -   Git commit hash for reproducibility.
        -   Python environment details (key package versions).
-   **Learning Rate Scheduler State:**
    -   [ ] Ensure the state of the learning rate scheduler is also saved and loaded if one is used.

## II. General Project Enhancements

-   **Configuration Management:**
    -   [ ] Implement a centralized configuration system (e.g., using YAML or JSON files) to manage hyperparameters, model configurations, dataset paths, and other settings. This will make experiments easier to track, reproduce, and modify.
-   **Experiment Tracking & Visualization:**
    -   [ ] Integrate with experiment tracking tools like TensorBoard, Weights & Biases, or MLflow.
    -   [ ] Log metrics, hyperparameters, model architecture, and sample predictions.
    -   [ ] Enhance `training-visualizer.py` or replace/supplement it with these tools.
-   **Testing Framework:**
    -   [ ] Create a `tests/` directory.
    -   [ ] Implement unit tests for critical components:
        -   Data loading and preprocessing (`data/`).
        -   Model components and architecture (`components/`).
        -   Loss functions (`components/loss.py`).
        -   Checkpointing logic (`util/checkpoint_manager.py`).
        -   Utility functions (`util/util.py`).
-   **Data Handling (`data/`):**
    -   [ ] Review and potentially expand data augmentation techniques relevant to medical imaging.
    -   [ ] Ensure robust preprocessing steps and normalization.
    -   [ ] Add checks for data integrity and consistency.
-   **Evaluation Metrics:**
    -   [ ] Implement a comprehensive set of evaluation metrics suitable for medical image segmentation (e.g., Dice Score, Intersection over Union (IoU), Hausdorff Distance, Precision, Recall, Specificity, F1-score).
    -   [ ] Ensure metrics are calculated correctly and reported clearly during training and testing.
-   **Training Loop (`train-test/train.py`):**
    -   [ ] **Early Stopping:** Implement an early stopping mechanism to prevent overfitting by monitoring a validation metric.
    -   [ ] **Learning Rate Scheduling:** Ensure a learning rate scheduler is used and its state is properly managed with checkpoints.
    -   [ ] **Gradient Clipping:** Consider adding gradient clipping for training stability, if necessary.
    -   [ ] **Reproducibility:** Ensure all random seeds (PyTorch, NumPy, Python `random`) are set for reproducible results.
    -   [ ] **Mixed Precision Training:** Explore using mixed precision (e.g., `torch.cuda.amp`) for faster training and reduced memory usage if applicable.
-   **Inference/Prediction Script:**
    -   [ ] Create a dedicated script for running inference on new images or a test set.
    -   [ ] This script should handle loading a trained model, preprocessing input, running the model, and postprocessing the output (e.g., generating segmentation masks).
    -   [ ] Option to save or visualize predictions.
-   **Documentation:**
    -   [ ] **Project README:** Create/enhance `README.md` with:
        -   Project overview and goals.
        -   Setup instructions (dependencies, environment).
        -   Instructions on how to train the model.
        -   Instructions on how to run inference.
        -   Dataset preparation guidelines.
        -   Expected results or benchmarks.
    -   [ ] **Docstrings:** Add comprehensive docstrings to all modules, classes, and functions, explaining their purpose, arguments, and return values.
    -   [ ] **Code Comments:** Add inline comments to explain complex or non-obvious parts of the code.
-   **Dependency Management:**
    -   [ ] Create a `requirements.txt` file (using `pip freeze > requirements.txt`) or an `environment.yml` file (for Conda environments) to list all project dependencies and their versions.
-   **Code Quality & Style:**
    -   [ ] Integrate code formatting tools (e.g., Black, autopep8) to ensure consistent code style.
    -   [ ] Use linting tools (e.g., Flake8, Pylint) to identify potential errors and style issues.
    -   [ ] Set up pre-commit hooks to automate formatting and linting.
-   **Distributed Training Support:**
    -   [ ] For future scalability with larger datasets or models, consider adding support for distributed training (e.g., using `torch.nn.DataParallel` or `torch.nn.parallel.DistributedDataParallel`).
-   **Input Validation:**
    -   [ ] Add validation for function arguments and configuration parameters to catch errors early.
-   **Model Export/Deployment:**
    -   [ ] Consider adding functionality to export the trained model to formats suitable for deployment (e.g., ONNX, TorchScript).

## III. Model Specific (`components/`)

-   **Ablation Studies:**
    -   [ ] Plan and conduct ablation studies to understand the contribution of different model components (e.g., `feat-guided-unet`, `medclip-unet`, `trans-de-fusion`).
-   **Hyperparameter Tuning:**
    -   [ ] Implement a systematic approach for hyperparameter tuning (e.g., grid search, random search, Bayesian optimization).

This list can be used as a checklist to track progress and prioritize future development efforts.
