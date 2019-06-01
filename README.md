# tvmodels
The tvmodels library contains implementation of pretrained vision models in pytorch trained on ImageNet. Some of these models are not available in [torchvision](https://pytorch.org/docs/stable/torchvision/index.html), so you can load them for here.

## Installation
Run the following to install:
```python
pip install tvmodels
```

## Usage
```python
from tvmodels.models import se_resnet50, resnet18

# Load the models
se_res_model = se_resnet50(pretrained=True)
res_model = resnet18(pretrained=True)
```

## TODO
- [x] SEResNet50
- [x] SEResNet101
- [x] SEResNet152
- [x] SEResNeXt50_32_4
- [x] SEResNeXt101_32_4
- [x] SENet154
- [x] ResNext50_32_4
- [x] ResNext101_32_4
- [x] ResNext101_32_8
- [x] ResNext101_64_4
- [ ] ResNext152
- [x] ResNet18
- [x] ResNet34
- [x] ResNet50
- [x] ResNet101
- [x] ResNet152
