## My version of InceptionResNetV2 with larger input image

import torch.utils.model_zoo as model_zoo
from pretrainedmodels.models.inceptionresnetv2 import InceptionResNetV2, pretrained_settings
import torch.nn as nn

class MyInceptionResNetV2(InceptionResNetV2):
    def __init__(self, *args, **kwargs):
        super(MyInceptionResNetV2, self).__init__(*args, **kwargs)
        self.conv2d_last = nn.Conv2d(1536, 28, 1)
        self.avgpool_last = nn.AvgPool2d(8, count_include_pad=False)
    
    def logits(self, features):
        x = self.avgpool_1a(features)
        #x = x.view(x.size(0), -1)
        #x = self.last_linear(x)
        x = self.conv2d_last(x)
        x = x.view(x.size(0), -1)
        return x
    
def inceptionresnetv2(num_classes=1000, pretrained='imagenet'):
    r"""InceptionResNetV2 model architecture from the
    `"InceptionV4, Inception-ResNet..." <https://arxiv.org/abs/1602.07261>`_ paper.
    """
    if pretrained:
        settings = pretrained_settings['inceptionresnetv2'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        # both 'imagenet'&'imagenet+background' are loaded from same parameters
        model = MyInceptionResNetV2(num_classes=1001)
        model.load_state_dict(model_zoo.load_url(settings['url']), strict=False)

#         if pretrained == 'imagenet':
#             new_last_linear = nn.Linear(1536, 1000)
#             new_last_linear.weight.data = model.last_linear.weight.data[1:]
#             new_last_linear.bias.data = model.last_linear.bias.data[1:]
#             model.last_linear = new_last_linear

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']

        model.mean = settings['mean']
        model.std = settings['std']
    else:
        model = MyInceptionResNetV2(num_classes=num_classes)
    return model


class TransferedModel(nn.Module):
    
    def __init__(self,
                 pretrained,
                 num_classes):
        super(TransferedModel, self).__init__()
        self.pretrained = pretrained
        n_feature = pretrained.last_linear.in_features
        self.classifier = nn.Sequential(
            #nn.Linear(n_feature, n_feature),
            #nn.BatchNorm1d(n_feature),
            #nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv1d(16, num_classes, kernel_size=1))
        self.pretrained.last_linear = self.classifier
    
    
    def forward(self, x):
        x = self.pretrained(x)
        return x
        
#             nn.Conv2d(n_feature, n_feature, 1),
#             nn.ReLU(inplace=True),