<rewrite_this>
1. Introduction
    - Background and Motivation
    - Problem Statement
    - Contributions

2. Related Work
    - Hyperparameter Optimization (HPO)
    - Freeze-Thaw Bayesian Optimization
    - Prior-Data Fitted Networks (PFNs)

3. Methodology
  - Overview: how the things are connected
    - Stage 1: Single Transformer Training
        - Data Preparation
        - Model Architecture
        - Training Procedure
    - Stage 2: Training with Synthetic Multi-Curves
        - Data Preparation
        - Model Architecture
        - Training Procedure
    - Stage 3: Training with Real Curves
        - Data Preparation
        - Model Architecture
        - Training Procedure

4. Experimental Setup
    - Synthetic Data Experiments
    - Multi-Curve Synthetic Data Experiments
    - Real Data Experiments
    - Baseline Comparisons
    - Robustness and Generalization Experiments
    - Fine-Tuning and Adaptation Experiments

5. Results and Discussion
    - Prediction Accuracy
    - HPO Performance
    - Ablation Study
    - Weighing Scheme Comparison
    - Context Size Variation
    - Noise Robustness
    - Hyperparameter Sensitivity
    - Generalization Across Tasks
    - Fine-Tuning and Adaptation

6. Conclusion
    - Summary of Findings
    - Future Work
</rewrite_this>


## What is missing from the ifbo paper?

The number of hyperparameters that is supported is limited by at most 10, the number of learning curves is limited by at most $tokens / horizon$, the optimization process is pretty slow.
