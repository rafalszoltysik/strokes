# Stroke prediction (university project)

Coursework project: a small Python pipeline for **stroke risk prediction** on the public [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (`healthcare-dataset-stroke-data.csv`).

This is an academic exercise — not a medical product and not clinical advice.

## Stack

- Python 3
- pandas, NumPy, scikit-learn
- imbalanced-learn
- matplotlib / seaborn
- PyYAML, joblib

## Layout

```
main.py           # entrypoint
config/           # pipeline config
data/raw/         # dataset CSV
data/processed/   # intermediate outputs
results/          # reports / artefacts
src/              # analysis, ML, pipeline code
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

Ensure the dataset is at:

`data/raw/healthcare-dataset-stroke-data.csv`

(Download from Kaggle if you clone without the CSV.)

## Run

```bash
python main.py
```

Logs report success/failure and the path to the generated report under `results/`.

## License

MIT — see [LICENSE](./LICENSE).

The dataset is third-party; follow its own license / terms on Kaggle.
