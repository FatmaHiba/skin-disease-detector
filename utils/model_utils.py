import timm
import torch.nn as nn

config = {
    'dropout_rate': 0.3
}

def create_model(num_classes):
    model = timm.create_model("efficientnet_b0", pretrained=False, drop_rate=config['dropout_rate'])
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.SiLU(),
        nn.Dropout(config['dropout_rate']),
        nn.Linear(512, num_classes)
    )
    return model