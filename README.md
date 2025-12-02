CIFAR-10 Image Classification using CNNs



CIS 483 Term Project - Computer Vision

Team Members

\- Jeneen Jadallah 

&nbsp;-Haneen Yahfoufi

&nbsp;-Solomiya Pylypiv

Course: CIS 483 - Deep Learning  

Instructor\*\*: Dr. John P. Baugh  

Project Overview



This project implements and compares two Convolutional Neural Network (CNN) architectures for image classification on the CIFAR-10 dataset. We developed a baseline CNN and an improved variant incorporating batch normalization, deeper architecture, and enhanced regularization techniques.



Objectives

1\. Implement a baseline CNN for image classification

2\. Develop an improved CNN architecture with modern techniques

3\. Compare performance between both models

4\. Analyze results and provide insights into architectural design choices



Key Results

\- Baseline CNN: 81.41% test accuracy

\- Improved CNN: 89.54% test accuracy

\- Improvement: +8.13 percentage points



Dataset: CIFAR-10



The CIFAR-10 dataset consists of 60,000 32×32 color images in 10 classes:



| Property | Value |

|----------|-------|

| Training samples | 50,000 |

| Test samples | 10,000 |

| Image size | 32×32×3 (RGB) |

| Number of classes | 10 |

| Class distribution | Balanced (6,000 per class) |



Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck



Data Preprocessing

\- Normalization: Images normalized using CIFAR-10 mean and standard deviation

&nbsp; - Mean: \[0.4914, 0.4822, 0.4465]

&nbsp; - Std: \[0.2023, 0.1994, 0.2010]

\- Data Augmentation (Training only):

&nbsp; - Random horizontal flips (50% probability)

&nbsp; - Random crops with 4-pixel padding

\- No augmentation applied to test set to ensure fair evaluation



Model Architectures



1\. Baseline CNN



A simple 3-layer convolutional network serving as our baseline:



Architecture:

INPUT (32×32×3)

&nbsp;   ↓

CONV(32 filters, 3×3) → ReLU → MaxPool(2×2)

&nbsp;   ↓

CONV(64 filters, 3×3) → ReLU → MaxPool(2×2)

&nbsp;   ↓

CONV(128 filters, 3×3) → ReLU → MaxPool(2×2)

&nbsp;   ↓

FLATTEN → FC(256) → ReLU → Dropout(0.5) → FC(10)

&nbsp;   ↓

OUTPUT (10 classes)



Specifications:

\- Total parameters: ~1.2 million

\- Dropout rate: 0.5

\- Activation: ReLU

\- Pooling: Max pooling (2×2)



2\. Improved CNN



An enhanced architecture incorporating modern deep learning techniques:



Architecture:

INPUT (32×32×3)

&nbsp;   ↓

CONV(64, 3×3) → BatchNorm → ReLU

CONV(64, 3×3) → BatchNorm → ReLU → MaxPool → Dropout(0.2)

&nbsp;   ↓

CONV(128, 3×3) → BatchNorm → ReLU

CONV(128, 3×3) → BatchNorm → ReLU → MaxPool → Dropout(0.3)

&nbsp;   ↓

CONV(256, 3×3) → BatchNorm → ReLU

CONV(256, 3×3) → BatchNorm → ReLU → MaxPool → Dropout(0.4)

&nbsp;   ↓

FLATTEN → FC(512) → BatchNorm → ReLU → Dropout(0.5) → FC(10)

&nbsp;   ↓

OUTPUT (10 classes)



Specifications:

\- Total parameters: ~5.8 million

\- Batch normalization after each convolutional and FC layer

\- Progressive dropout: 0.2 → 0.3 → 0.4 → 0.5

\- Deeper architecture: 6 convolutional layers vs 3



Key Improvements:

1\. Batch Normalization: Stabilizes training, enables faster convergence

2\. Increased Depth: 6 conv layers for more complex feature learning

3\. Progressive Dropout: Stronger regularization in deeper layers

4\. More Filters: 64 → 128 → 256 for richer feature representations



Training Configuration



Hyperparameters

Optimizer:         Adam

Learning Rate:     0.001

Weight Decay:      1e-4 (L2 regularization)

Batch Size:        128

Epochs:            30

Loss Function:     Cross-Entropy Loss

Device:            CPU



Training Strategy

\- Adam optimizer with default β parameters (β₁=0.9, β₂=0.999)

\- L2 weight regularization (1e-4) to prevent overfitting

\- Cross-entropy loss for multi-class classification

\- Best model saved based on test accuracy



Results



Performance Comparison



| Model | Train Accuracy | Test Accuracy | Parameters | Training Time |

|-------|----------------|---------------|------------|---------------|

| Baseline CNN | ~78% | 81.41% | 1.2M | ~10 min |

| Improved CNN | ~90% | 89.54% | 5.8M | ~15 min |

| Improvement | +12% | +8.13% | +4.6M | +50% |



Key Findings



1\. Significant Performance Gain: The improved model achieved 8.13 percentage points higher accuracy than the baseline



2.Faster Convergence: Batch normalization enabled the improved model to reach high accuracy earlier in training



3\. Better Generalization: Despite having 4.8× more parameters, the improved model showed better generalization with a smaller train-test accuracy gap



4\. Overfitting Control: Progressive dropout and batch normalization effectively prevented overfitting in the deeper network



Training Dynamics



!\[Training Results](results.png)



Left Graph - Training Progress:

\- Improved model (red/green) converges faster than baseline (blue/orange)

\- Batch normalization accelerates learning in early epochs

\- Both models show steady improvement throughout 30 epochs

\- Test accuracy curves remain close to training curves, indicating good generalization



Right Graph - Final Comparison:

\- Clear visual demonstration of 8.13% improvement

\- Both models exceed 80% accuracy threshold

\- Improved model approaches 90% accuracy mark



Installation \& Setup



Prerequisites

\- Python 3.8 or higher

\- pip package manager

\- (Optional) CUDA-capable GPU for faster training



Step 1: Clone Repository

bash

git clone https://github.com/yourusername/cifar10-project.git

cd cifar10-project



Step 2: Install Dependencies

bash

pip install torch torchvision matplotlib numpy tqdm scikit-learn



Or using requirements.txt:

bash

pip install -r requirements.txt



Step 3: Run Training

bash

python main.py



What happens:

1\. CIFAR-10 dataset downloads automatically (~170MB)

2\. Baseline CNN trains for 30 epochs

3\. Improved CNN trains for 30 epochs

4\. Results visualization saved as `results.png`

5\. Best models saved as `.pth` files



Expected Runtime

\- CPU: ~25-30 minutes total

\- GPU: ~8-12 minutes total

Project Structure



cifar10-project/

│

├── main.py                     # Main training script

├── README.md                   # This file

├── requirements.txt            # Python dependencies

├── results.png                 # Training results visualization

├── report.pdf                  # Project report (6-8 pages)

├── presentation.pptx           # Presentation slides

│

├── data/                       # CIFAR-10 dataset (auto-downloaded)

│   └── cifar-10-batches-py/

│

├── Baseline\_best.pth          # Saved baseline model weights

└── Improved\_best.pth          # Saved improved model weights



Usage Examples



Basic Training

bash

Train both models with default settings

python main.py



Loading Saved Models

python

import torch

from main import ImprovedCNN



Load trained model

model = ImprovedCNN()

model.load\_state\_dict(torch.load('Improved\_best.pth'))

model.eval()



Make predictions

with torch.no\_grad():

&nbsp;   predictions = model(input\_images)

&nbsp;   predicted\_classes = predictions.argmax(dim=1)



Customizing Training

Modify these parameters in `main.py`:



python

Training configuration

epochs = 50              # Increase for better results

batch\_size = 64          # Reduce if out of memory

learning\_rate = 0.001    # Adjust learning rate



Model selection

train\_model(baseline, "Baseline", epochs=epochs)

train\_model(improved, "Improved", epochs=epochs)



Technical Analysis



Why Did the Improved Model Perform Better?



1\. Batch Normalization

&nbsp;  - Normalized activations reduced internal covariate shift

&nbsp;  - Enabled higher learning rates and faster convergence

&nbsp;  - Acted as regularizer, improving generalization



2\. Deeper Architecture

&nbsp;  - More layers enabled hierarchical feature learning

&nbsp;  - Early layers learned edges and textures

&nbsp;  - Deeper layers learned complex object parts

&nbsp;  - 6 layers vs 3 provided more representational capacity



3\. Progressive Dropout

&nbsp;  - Increasing dropout rates (0.2 → 0.4) in deeper layers

&nbsp;  - Prevented co-adaptation of neurons

&nbsp;  - Reduced overfitting without sacrificing capacity



4\. Increased Model Capacity

&nbsp;  - More filters (64 → 128 → 256) captured richer features

&nbsp;  - 5.8M parameters vs 1.2M allowed more complex decision boundaries



Challenges Encountered



1\. Training Time: Improved model took 50% longer to train

2\. Overfitting Risk: Mitigated with dropout and data augmentation

3\. Hyperparameter Tuning: Required experimentation to find optimal settings

4\. Memory Constraints: Deeper model required more GPU/RAM



Comparison to State-of-the-Art



| Approach | Test Accuracy | Notes |

|----------|---------------|-------|

| \*\*Our Baseline\*\* | 81.41% | Simple 3-layer CNN |

| \*\*Our Improved\*\* | \*\*89.54%\*\* | 6-layer CNN with BatchNorm |

| ResNet-110 | ~93% | Residual connections |

| DenseNet-BC | ~96% | Dense connections |

| Vision Transformer | ~99%+ | Attention mechanisms |





References



1\. A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images," Technical Report, 2009.



2\. A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks," NIPS, 2012.



3\. K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," CVPR, 2016.



4\. S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift," ICML, 2015.



5\. N. Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting," JMLR, 2014.



6\. K. Simonyan and A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," ICLR, 2015.



7\. I. Goodfellow, Y. Bengio, and A. Courville, "Deep Learning," MIT Press, 2016.



8\. Y. LeCun et al., "Gradient-Based Learning Applied to Document Recognition," Proceedings of the IEEE, 1998.



