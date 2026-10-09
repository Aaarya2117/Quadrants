# Tech Stack: ML Model Swap

Overview of the tools, libraries, and runtime environments powering `modelswap`.

---

## 1. Core Programming Environment
- **Python 3.11+ / 3.13**: Primary language for the core runtime, comparison engine, CLI, and model execution.
- **Operating Systems**: Cross-platform (Windows, Linux, macOS).

---

## 2. Machine Learning Frameworks
- **PyTorch (`torch`)**:
  - Neural network definition, checkpoint serialization (`.pt`), and evaluation.
  - CUDA GPU acceleration enabled for transformer inference and fine-tuning.
  - Fast tensor matrix operations for parameter divergence ($\Delta W$ Frobenius norms).
- **Hugging Face Transformers (`transformers`)**:
  - Pre-trained and fine-tuned architectures (`bert-base-uncased`, `roberta-base`).
  - AutoTokenizer & AutoModelForSequenceClassification interfaces.
- **Scikit-Learn & NumPy**:
  - Synthetic dataset generation (`make_moons`) and standardization.
  - Evaluation metrics: accuracy, cross-entropy loss, and percentile latency timing.

---

## 3. Configuration & Swap Engine
- **PyYAML (`yaml`)**:
  - Model registry configuration (`models.yaml`) parsing and serialization.
- **Python Standard Library (`os`, `shutil`, `tempfile`, `hashlib`, `time`)**:
  - Atomic file replacement via `os.replace` (POSIX `rename(2)` semantics).
  - Checksum hashing (SHA256) for parameter state integrity.
  - Automatic backup restoration on failure (`models.yaml.bak`).
  - Structured audit trail written to `swaps/<timestamp>_<role>.json`.

---

## 4. CLI & Interactive Experience
- **Argparse & Color-formatted Terminal Output**:
  - CLI commands (`status`, `compare`, `apply`, `rollback`, `reset`, `demo`).
  - Interactive console menu with model pair selection and interactive action prompts.
- **Pytest**:
  - 105+ unit and integration tests covering architecture checks, comparison logic, engine operations, weight diffing, and CLI interactions.
