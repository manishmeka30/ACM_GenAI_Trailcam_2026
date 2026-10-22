# Trailcam Project - Classification of Animal and Vehicle Images (ACM GenAI Task 2026)

**Google Colab Link:** https://colab.research.google.com/drive/1gJuYbPnFpDP-gzGtap4Jvi70gTc7iqxW?usp=sharing

## 1. CNN Architecture
A custom CNN architecture consisting of two convolution blocks (`TrailcamCNN`) was developed using PyTorch for the 10 classes from CIFAR-10 dataset ($32 \times 32$ RGB images):
- **Conv Block 1:** `Conv2d(3 -> 32, kernel=3x3, padding=1)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2x2)` (Output: $32 \times 16 \times 16$)
- **Conv Block 2:** `Conv2d(32 -> 64, kernel=3x3, padding=1)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2x2)` (Output: $64 \times 8 \times 8$)
- **Classifier Head:** `Flatten` ($4096$ features) $\rightarrow$ `Linear(4096 -> 256)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(256 -> 10)`
- **Training Hyperparams:** Both experiments had the same parameters: 10 epochs, batch size 64, `CrossEntropyLoss`, and `Adam` optimizer (`lr = 0.001`).

## 2. Comparing the Final Accuracy
| Experiment | Data Augmentation Used | Final Training Accuracy | Final Test (Validation) Accuracy |
| :--- | :--- | :--- | :--- |
| **Run A** | No data augmentation (Normalized only) | `96.98%` | **`72.51%`** |
| **Run B** | `RandomHorizontalFlip(0.5)` + `RandomCrop(32, pad=4)` | `73.07%` | **`74.35%`** |

## 3. Field Report for the Forest Department
**Was data augmentation effective?**
Yes. During the 10-epoch training period, the **unmodified** **Run A** managed to achieve high accuracy during training (~97%), but its validation loss increased sharply past epoch 4, 
which indicated **overfitting**. Run B with random flips and crops, on the other hand, added regularization to the training process and achieved higher test accuracy (**74.35%**)
without overfitting and being more suitable for actual camera trap video footage where animals could be shown at various angles.

**Which two classes are the hardest for the model to differentiate?**
As per the confusion matrix, the classes that the model most commonly mixes up are **`cat` and `dog`** (mixes cats up with dogs 261 times). At such a small $32 \times 32$ resolution, 
cats and dogs have very similar features (texture, limbs, environment).

## 4. Usage of AI
Utilized AI in structuring the pipeline of PyTorch, loop training, and evaluation criteria while strictly maintaining the random seed for reproducing results from Run A to Run B (`seed=42`).
