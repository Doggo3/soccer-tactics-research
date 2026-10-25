# Classifying soccer attacking phases

Streamlit research site based on Jack Chiang's supplied paper, 22-cell notebook, and 2,955-row SkillCorner phase-summary dataset.

## What the site does
- Explains the five attacking-phase labels and the match-based evaluation.
- Classifies actual held-out phases using all 18 recorded features.
- Filters and visualizes the supplied records, with scoped CSV downloads.
- Presents the paper/notebook's saved confusion matrix, model comparisons, importance scores, and grouped validation.

No player simulations, synthetic trajectories, editable formations, or movement predictions are included.

## Run

```sh
pip install -r requirements.txt
streamlit run app.py
```

## Evidence and reproduction

`data.csv` is the supplied `skillcorner_enhanced_dataset(1).csv`. Matches 2015213 and 1899585 form the 567-phase test split; the other 2,388 phases are training data. `train.py` fits the supplied notebook's Enhanced XGBoost configuration. The live fitted model is `model.ubj`, with metrics and input order in `metadata.json`.

The original notebook contains saved results but no historical fitted model or pinned dependencies. Rerunning its training cells in the site environment yields the exact predictions of `model.ubj`: accuracy 0.9206349, macro F1 0.8754636. The notebook's saved historical results are accuracy 0.919 and macro F1 0.864. The site distinguishes these two runs. It does not claim to reproduce the historical fitted model exactly.

`paper_results.json` transcribes the saved notebook's confusion matrix (cell 6), feature importance (cell 8), and fold scores (cell 16). The confusion matrix has 567 entries and 521 correct labels. The grouped validation used a 300-tree Random Forest. The tuning settings summarized in cell 18 are distinct from the XGBoost fitted in cell 4.

## Limits

The CSV contains phase summaries, not full trajectories or video. The original extraction pipeline is not included. Reference labels come from SkillCorner's system. Ten matches from one league do not establish cross-league or coach-label performance. Some stored progress-speed values require extraction review before physical interpretation.
