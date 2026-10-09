# Tech Stack: ML Model Swap (3-Hour Sprint)

Overview of the tools, libraries, and runtime environments used to build and demonstrate ML Model Swap.

---

## 1. Programming Languages & Runtimes
- **Python 3.11+**: Primary language for the core runtime, comparison engine, CLI, and model execution.
- **MATLAB (R2023b / R2024a or GNU Octave)**: Used for interactive simulation, 2D decision boundary rendering, weight divergence heatmaps, and live data stream visualization.

---

## 2. Machine Learning Frameworks
- **PyTorch / TorchScript (`torch`)**:
  - Defines and executes the 2-layer MLP neural network architecture.
  - Loads and evaluates weights for Model A (`model_a.pt`) and Model B (`model_b.pt`).
- **Scikit-Learn & NumPy**:
  - Generates synthetic 2D classification datasets (e.g., `make_moons` or `make_classification`).
  - Evaluates performance metrics (Accuracy, F1-Score, Confusion Matrix, Latency).

---

## 3. Configuration & Swap Engine
- **PyYAML**: Parses and atomically updates the model registry (`models.yaml`).
- **Pydantic (v2)**: Validates model schemas, tensor dimensions, and contract constraints.
- **Python Standard Library (`os`, `shutil`, `json`, `time`)**:
  - Provides atomic file replacement via `os.replace` (POSIX `rename(2)`).
  - Handles backup restoration and writes audit records to `swaps/`.

---

## 4. CLI & Interface
- **Typer / Argparse**: Powers the lightweight command-line interface:
  - `modelswap compare`
  - `modelswap apply`
  - `modelswap rollback`

---

## 5. MATLAB Simulation Stack
- **MATLAB Plotting & Graphics (`plot`, `contourf`, `imagesc`, `subplot`)**:
  - Renders side-by-side decision boundary surfaces for Model A and Model B.
  - Visualizes weight tensor difference: $\Delta W = W_B - W_A$.
  - Animates a real-time stream of predictions demonstrating seamless handover during the swap event.
