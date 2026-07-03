# Distracted Driver Detection — Deep Learning (Machine Vision)

Image-classification project that detects **10 categories of distracted driving** (safe driving, texting, phone, radio, reaching behind, etc.) from in-car camera images, benchmarking custom networks against transfer learning. Course: *Machine Vision* (CSE480), Ain Shams University.

**▶ Notebook:** [`distracted_driver_detection.ipynb`](distracted_driver_detection.ipynb) (in this repo, with output visualizations) · [Google Colab](https://colab.research.google.com/drive/1trv1lqV-g_q12Vy5fr9ZKNy58sWPGzxT)

## Dataset
State Farm Distracted Driver Detection (Kaggle) — in-car images labeled `c0`–`c9`.

## Approach
- **Preprocessing:** cleaning/dedup, resize to 224×224, normalization, RGB retained.
- **Augmentation:** rotation (±15°), horizontal flip, zoom, brightness, width/height shift.
- **Split:** 70% train / 20% val / 10% test, batched via `ImageDataGenerator`.
- **Models compared (5):** baseline **DNN**, custom **CNN**, and transfer learning with **ResNet50**, **VGG16**, **MobileNet**.
- **Training:** Adam, categorical cross-entropy, EarlyStopping + ModelCheckpoint.
- **Evaluation:** accuracy, precision, recall, F1, confusion matrices.

## Results
| Model | Val accuracy | Note |
|---|---|---|
| **ResNet50** | **~92%** | best — strong generalization |
| VGG16 | ~90% | accurate but heavy |
| MobileNet | ~88% | best speed/accuracy trade-off → deployment pick |
| CNN (custom) | ~78% | beats DNN, mild overfitting |
| DNN (baseline) | ~65% | no spatial features |

**Takeaway:** transfer learning clearly beat custom nets; MobileNet is the practical choice for in-vehicle embedded deployment.

## Tech stack
Python · TensorFlow / Keras · NumPy · Matplotlib · scikit-learn (metrics)

## Credits
Team project at Ain Shams University (Machine Vision / CSE480): Mohamed Refaie and Fares Mohamed.
