import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet152_Weights, DenseNet161_Weights, MobileNet_V2_Weights

# Function to select model
def select_model(model_name):
    if model_name == "resnet152":
        model = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 4) 
    elif model_name == "densenet161":
        model = models.densenet161(weights=DenseNet161_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, 4)  
    elif model_name == "mobilenet":
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4) 
    else:
        raise ValueError("Unknown model name. Choose from 'resnet152', 'densenet161', 'mobilenet'.")
    
    return model


if __name__ == "__main__":
    # Choose model
    model_names = ['resnet152', 'densenet161','mobilenet']
    for model_name in model_names:
        model = select_model(model_name)