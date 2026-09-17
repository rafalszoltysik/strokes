# strokes

University coursework: a Python pipeline for stroke risk prediction on the public [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).

Academic exercise only. Not a medical product and not clinical advice.

## What it does

Running `main.py` loads `data/raw/healthcare-dataset-stroke-data.csv`, then:

1. Cleans and preprocesses the data (settings in `config/config.yaml`)
2. Builds exploratory plots under `results/plots/`
3. Trains Random Forest, Gradient Boosting, Logistic Regression, and SVM
4. Evaluates models, handles class imbalance (SMOTE, SMOTETomek, or undersampling), tunes the classification threshold, and calibrates the best model
5. Writes a text report under `results/reports/`

## Requirements

- Python 3
- Packages from `requirements.txt`: pandas, NumPy, scikit-learn, imbalanced-learn, matplotlib, seaborn, PyYAML, joblib

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

The dataset CSV is already in `data/raw/healthcare-dataset-stroke-data.csv`. If you remove it, download it again from Kaggle and place it at that path.

## Usage

```bash
python main.py
```

Logs show success or failure and the path to the generated report.

Pipeline settings live in `config/config.yaml` (train/test split, imputation, model parameters, drift threshold).

## Layout

```
main.py
config/config.yaml
data/raw/
data/processed/
results/
src/                 # preprocessing, analysis, ML, pipeline
```

## License

MIT. See [LICENSE](./LICENSE).

The dataset is third-party. Follow its terms on Kaggle.
