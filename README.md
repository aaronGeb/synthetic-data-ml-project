# Synthetic Data ML Project

This repository contains a machine learning project focused on the generation, analysis, and modeling of synthetic data. The aim of this project is to explore various machine learning algorithms and techniques using synthetic datasets, ensuring a controlled environment to test different models, validate assumptions, and experiment with hyperparameter tuning.

## Project Overview

The project leverages synthetic data to simulate real-world scenarios and test various machine learning models, such as classification, regression, and clustering algorithms. It also includes preprocessing steps, model evaluations, and metrics collection.

### Key Features:
- **Synthetic Data Generation**: Scripts and methodologies used to generate synthetic datasets.
- **Data Preprocessing**: Preprocessing steps including handling missing values, feature scaling, and encoding.
- **Modeling**: Various machine learning models such as CatBoost, Random Forest, etc.
- **Hyperparameter Tuning**: Search and optimization of hyperparameters using techniques like Grid Search or Randomized Search.
- **Model Evaluation**: Evaluation metrics including accuracy, precision, recall, F1-score, ROC-AUC, etc.
- **Experiment Tracking**: Methods for tracking different experiments and results.

## Setup Instructions

### Prerequisites

Make sure you have the following libraries installed:

- Python 3.x
- `pip` for installing dependencies

### Installing Dependencies

Clone this repository:

```bash
git clone https://github.com/aaronGeb/synthetic-data-ml-project.git
```
```
cd synthetic-data-ml-project
```
### Install the required dependencies:
1.	Create a new Conda environment (if you don’t have one already):
```
conda create -n synthetic python=3.11
```
2.Activate the environment:
```
conda activate synthetic
```
3.Exports the environment’s configuration.
```
conda synthetic export >environmental.yml
```

### Model Evaluation

After training, the model will output evaluation metrics such as accuracy, precision, recall, and F1 score, among others. The results will be stored in model_results/.

### Contributing

If you would like to contribute to the project, feel free to fork this repository and create a pull request with your changes. Please ensure your code follows the project’s coding style and includes tests.

### License

This project is licensed under the [MIT License](LICENSE) for details.

### Acknowledgments
- CatBoost and XGBoost: A powerful gradient boosting library.
- Scikit-learn: A widely used machine learning library in Python.
- Synthetic Data: Used for testing models in a controlled and reproducible environment.
- Docker for containerization.