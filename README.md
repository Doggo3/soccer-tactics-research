# Soccer tactics classification

Can player and ball tracking measurements identify the type of attack? This project compares machine-learning models on 2,955 attacking phases from 10 Australian A-League matches, using SkillCorner data.

The five labels are **build-up, creation, direct attack, fast transition, and finishing**.

[Explore the app](https://machine-learning-soccer-attacking-phase-classification.streamlit.app/) · [Read the paper](https://machine-learning-soccer-attacking-phase-classification.streamlit.app/?section=paper)

## Results

The study reserves two entire matches for testing: 567 phases the models do not see during training.

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Enhanced XGBoost | 91.9% | 0.864 |
| Enhanced Random Forest | 91.2% | 0.873 |

XGBoost had the highest overall accuracy. Random Forest had the highest macro F1, which weights each category equally. Fast transitions were the hardest category: XGBoost identified 13 of 25 correctly.

These are the paper’s reported results. The app uses a rerun of the notebook’s XGBoost configuration, with slightly different scores. See [reproduction notes](docs/reproduction.md).

## Run locally

Use Python 3.12 and a virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Files

- `app.py` — classifier, dataset explorer, and results pages
- `paper_web.py` — the paper’s web edition
- `data.csv` — the 2,955 recorded phase summaries
- `train.py` — fit XGBoost and save the model and evaluation metrics
- `model.ubj`, `metadata.json` — fitted model, input order, and test results
- `paper_info.json`, `paper_results.json` — manuscript text and reported results
- `definitions.py` — measurement names, units, and explanations

## About the data

The target labels come from SkillCorner’s system. The models learn to reproduce those labels; they do not predict future player movement. Ten matches from one league also limit how far the results can be generalized.

Research by Jack Chiang, with guidance from Omar Ramos Escoto and support from Inspirit AI. Tracking data provided by [SkillCorner Open Data](https://github.com/SkillCorner/opendata).
