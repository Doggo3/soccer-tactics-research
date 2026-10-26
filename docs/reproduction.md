# Reproducing the model

The source notebook is *Classifying Soccer Tactics*. Its training cells use the supplied `skillcorner_enhanced_dataset(1).csv`, stored here as `data.csv`.

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
| Original notebook output | 0.919 | 0.864 |
| Model included in this repository | 0.920635 | 0.875464 |

Rerunning the notebook’s training cells with this repository’s dependencies produces the same predictions as the included model. The original notebook did not include its fitted model or pin dependency versions, so the cause of the difference from its saved output has not been established.

The reported search settings—depth 7 and learning rate 0.03—belong to a separate tuning summary. They are not the configuration fitted in the notebook’s main training cell.

## Paper results

`paper_results.json` records the original notebook’s saved outputs:

- Cell 6: confusion matrix, with 521 correct labels out of 567 phases.
- Cell 8: XGBoost feature importance.
- Cell 16: five-fold match-group validation, using a 300-tree Random Forest.

The [original manuscript](paper.pdf) includes assignment-template guidance. The site presents a condensed web edition of the research.

## Data limitations

The CSV contains phase summaries rather than full tracking sequences. The original feature-extraction pipeline was not included. Some signed progress-speed values are extreme and should not be interpreted as physical ball speed without checking that extraction. The definitions of spacing, wide zones, and player counts also cannot be reconstructed fully from the training notebook alone.
