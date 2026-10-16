# Pneumonia screening - holdout evaluation

Model: resnet18 (ImageNet init; stem+layer1+layer2 frozen; layer3+layer4+head fine-tuned), 224px input, best epoch 5 (val AUROC 0.9969).
Holdout set: 873 images from 520 patients (237 normal / 636 pneumonia), patient-disjoint from the other splits.

AUROC 0.9966 (95% CI 0.994-0.999) | AUPRC 0.9987

| Metric | threshold 0.5 | val-tuned threshold 0.920 | 95% CI (tuned) |
|---|---|---|---|
| Accuracy | 97.3% | 96.6% | 95.3% - 97.7% |
| Sensitivity (recall) | 99.1% | 96.4% | 94.8% - 97.7% |
| Specificity | 92.4% | 97.0% | 94.6% - 99.1% |
| PPV (precision) | 97.2% | 98.9% | |
| NPV | 97.3% | 90.9% | |
| F1 | 98.1% | 97.6% | |

Confusion matrix at tuned threshold (rows = truth, cols = predicted):

| | pred NORMAL | pred PNEUMONIA |
|---|---|---|
| true NORMAL | 230 | 7 |
| true PNEUMONIA | 23 | 613 |

Sensitivity by pneumonia subtype (tuned threshold): bacteria 97.7% (n=398), virus 94.1% (n=238)

By original dataset folder: test: n=96, AUROC 0.9931, acc 93.8%, train: n=777, AUROC 0.9990, acc 96.9%

CPU inference: 31 ms/image (batched).
