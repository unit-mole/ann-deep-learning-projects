# Evaluation Outputs

The committed outputs describe the supplied pretrained model evaluated on the
held-out test split at a 0.50 threshold.

- `model_metrics.json`: complete numeric audit
- `classification_report.csv`: class-level precision, recall, and F1
- `threshold_analysis.csv`: threshold-dependent test metrics for analysis
- `confusion_matrix.png`: true/false positive and negative counts
- `roc_curve.png`: ranking performance across thresholds
- `precision_recall_curve.png`: minority-class performance across thresholds
- `class_distribution.png`: source imbalance
- `model_architecture.png`: ANN layer diagram

Retraining through `src/model_training.py` will update the main metric and plot
files with results from the newly trained model.
