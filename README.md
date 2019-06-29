# tvmodels
The tvmodels library contains pretrained vision models in pytorch trained on ImageNet. Some of these models are available in [torchvision](https://pytorch.org/docs/stable/torchvision/index.html) but some are not, so you can load them for here.

## Installation
Run the following to install:
```python
pip install tvmodels
```

### Colab
If it shows `ModuleNotFoundError` on Google-colab use the following:
```python
!git clone https://github.com/rohitgr7/tvmodels.git
import sys
sys.path.append('/content/tvmodels')
```

## Usage
```python
from tvmodels.models import se_resnet50, resnet18

# Load the models
se_res_model = se_resnet50(pretrained=True)
res_model = resnet18(pretrained=True)
```

## Available models
- [x] ResNet(s)
- [x] ResNext(s)
- [x] SEResNet(s)
- [x] SEResNeXt(s)
- [x] SENet154
- [x] EfficientNets
