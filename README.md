<div align="center">

# 📉 Computational Foundations of Regression: Stability & Scalability

**Notebook:** `CDM__3_Final.ipynb`    
**Dataset:** [Auto MPG Data Set (UCI Machine Learning Repository)](https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data)    
**Author:** AmirHossein Talebi Koohestani    
**University:** Amir Kabir University of Technology (Tehran Polytechnic)    
**Supervisor:** Dr. Mehdi Ghatee    
**TA:** Dr. Behnam Yousefimehr

</div>

---

## 📖 Project Overview

This project evaluates the **numerical stability** and **computational scalability** of various linear regression solvers. It investigates the trade-offs between analytical solutions (Normal Equations, SVD) and iterative optimization algorithms (Batch Gradient Descent, Stochastic Gradient Descent).

The study is divided into two distinct phases:
1.  **Stability Analysis:** Using synthetic data to demonstrate the impact of **collinearity** (singular matrices) on Normal Equations versus Singular Value Decomposition (SVD).
2.  **Scalability & Convergence:** Using the **Auto MPG dataset** to compare the convergence speed and noise profiles of BGD and SGD, followed by implementing Polynomial Regression to model non-linear relationships.

---

## 🚀 Installation & Setup

### Prerequisites

- **Python 3.8+**
- **pip** (Python package manager)
- **Jupyter Notebook** or **JupyterLab**

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/regression-stability-scalability.git](https://github.com/your-username/regression-stability-scalability.git)
    cd regression-stability-scalability
    ```

2.  **Install required dependencies:**
    ```bash
    pip install numpy pandas scikit-learn matplotlib jupyter
    ```

3.  **Dataset Configuration:**
    The notebook is configured to automatically download the dataset from the UCI repository.
    - **Source URL:** `https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data`

4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

5.  **Open and run:**
    - Navigate to `CDM__3_Final.ipynb`
    - Run all cells sequentially

---

## 🎯 Objectives

- **Analyze Numerical Instability:** Demonstrate how Normal Equations fail or produce erratic weights when features are highly collinear (singular matrix), compared to the robustness of the Pseudoinverse (SVD).
- **Compare Optimization Algorithms:** Contrast Batch Gradient Descent (BGD) and Stochastic Gradient Descent (SGD) regarding:
    - Convergence speed (iterations vs. time).
    - Stability of the loss curve.
- **Implement Non-Linear Modeling:** Apply feature engineering (Polynomial Regression) to capture non-linear trends in the Auto MPG dataset.

---

## 🧠 Algorithms Implemented

| Algorithm | Type | Description |
|:---|:---|:---|
| **Normal Equations** | Analytical | Solves $\theta = (A^T A)^{-1} A^T y$. Fast for small data but unstable if $A$ is singular or features are collinear. |
| **SVD (Pseudoinverse)** | Numerical | Solves $\theta = A^+ y$ using Singular Value Decomposition. Handles collinearity robustly by managing near-zero singular values. |
| **BGD (Batch Gradient Descent)** | Iterative | Updates weights using the gradient of the *entire* dataset at every step. Smooth convergence but computationally expensive for large data. |
| **SGD (Stochastic Gradient Descent)** | Iterative | Updates weights using a single random training example per step. Faster iterations but "noisier" convergence path. |
| **Polynomial Regression** | Feature Eng. | Extends linear models by adding polynomial terms (e.g., $x^2$) to model curvature in data. |

---

## 🧩 Methodology

### Part 1: Synthetic Data & Stability
1.  **Data Generation:** Create a small synthetic dataset ($A$, $y$).
2.  **Baseline Solution:** Compute $\theta$ using Normal Equations, BGD, and SVD.
3.  **Collinearity Injection:** Introduce a "noisy duplicate" column to Matrix $A$ to create near-perfect collinearity.
4.  **Stress Test:** Compare how Normal Equations differ from SVD when the matrix approaches singularity. (Normal Equations produce exploded weights; SVD remains stable).

### Part 2: Auto MPG Analysis
1.  **Data Loading:** Fetch Auto MPG data from UCI, handle missing values (`?`), and drop incomplete rows.
2.  **Preprocessing:** Standardize features (Mean=0, Variance=1) using `StandardScaler` for optimal Gradient Descent performance.
3.  **BGD Implementation:** Run Batch Gradient Descent for 1000 iterations and log the cost.
4.  **SGD Implementation:** Run Stochastic Gradient Descent for 5000 iterations, resampling the history for comparison.
5.  **Polynomial Fit:** Transform the `Horsepower` feature into a 2nd-degree polynomial and train using SGD to fit a curve.

---

## 📊 Evaluation & Visualizations

The notebook produces the following key visualizations:

| Visualization | Description |
|:---|:---|
| **Convergence Comparison** | Overlays the cost history of BGD vs. SGD on a normalized iteration scale. Demonstrates BGD's smooth descent vs. SGD's stochastic fluctuations. |
| **Polynomial Regression Fit** | A scatter plot of `MPG` vs. `Horsepower` with the learned 2nd-degree polynomial curve (red line) overlayed, showing the non-linear relationship. |
