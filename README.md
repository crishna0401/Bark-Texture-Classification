# Bark-Texture-Images-Classification
Image classification task performed on BarkVN-50 dataset. Data consists of 50 categories of bark texture images. Total number is 5,578 images.

Download the Dataset on [kaggle](https://www.kaggle.com/datasets/saurabhshahane/barkvn50)

#multiclass image classification

Download best model weights from [link](https://drive.google.com/file/d/1BqqQIu9ZhTtS-Bt3AZaa5MhPupA8Rm23/view?usp=sharing) and keep in the same folder of the notebook



## Loading best model
```python
import torch
import torch.nn as nn
from torchvision import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load model
model = models.resnet50(weights=models. ResNet50_Weights.DEFAULT)

# Freeze model weights
for param in model.parameters():
    param.requires_grad = False

# Replace last layer for transfer learning
model.fc = nn.Sequential(
                      nn.Linear(model.fc.in_features, 256), 
                      nn.ReLU(), 
                      nn.Dropout(0.4),
                      nn.Linear(256, num_classes),                   
                      nn.LogSoftmax(dim=1))

model.load_state_dict(torch.load('model.ckpt'))
model.to(device)
model.eval() # comment this line if you are not evaluating
```

