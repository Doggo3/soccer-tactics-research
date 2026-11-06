# Reproducing the model

The project codebase, *Classifying Soccer Tactics*, contains the Python training and evaluation code and its saved results. It was developed in a Jupyter notebook. The code uses `skillcorner_enhanced_dataset(1).csv`, stored here as `data.csv`.

## Train/test split

- Training: 2,388 phases from eight matches.
- Testing: 567 phases from matches `2015213` and `1899585`.
- Inputs: 18 numerical features. Match ID, phase ID, and the target label are excluded.
- Target: one of the five SkillCorner attacking-phase labels.

Run `python train.py` to fit the model. This overwrites `model.ubj` and `metadata.json`.

The XGBoost configuration uses 400 trees, maximum depth 5, learning rate 0.05, row and column sampling of 0.8, balanced sample weights, and random seed 42. Dependencies are pinned in `requirements.txt`.

## Saved results and the rerun

| Run | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Original research run | 0.919 | 0.864 |
| Model included in this repository | 0.920635 | 0.875464 |

Rerunning the training code with this repository’s dependencies produces the same predictions as the included model. The original codebase did not include its fitted model or pin dependency versions, so the cause of the difference from its saved output has not been established.

The reported search settings (depth 7, learning rate 0.03) belong to a separate tuning summary. The main training code uses depth 5 and learning rate 0.05.

## Paper results

`paper_results.json` records the original codebase’s saved outputs:

- Confusion matrix, with 521 correct labels out of 567 phases.
- XGBoost feature importance.
- Five-fold match-group validation, using a 300-tree Random Forest.

The [manuscript](paper.pdf) is the author’s November 5, 2026 paper, preserved without edits. The site also includes a web summary.

## Data limitations

The CSV contains phase summaries rather than full tracking sequences. The original feature-extraction pipeline was not included. Some signed progress-speed values are extreme and should not be interpreted as physical ball speed without checking that extraction. The definitions of spacing, wide zones, and player counts also cannot be reconstructed fully from the training code alone.
