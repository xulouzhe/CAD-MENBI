import os
import torch
import numpy as np
import logging
from datetime import datetime
from collections import Counter
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.optim as optim

from dataset import CAD_MENBI_Dataset, prepare_dataset
from model import select_model
from utils import get_transforms, get_device

# Function to train and validate the model
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, fold=1, log_dir='log', weight_dir='weights', model_name='resnet152'):
    device = get_device()
    model = model.to(device)
    
    for epoch in range(num_epochs):
        logging.info(f"Fold {fold}, Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        logging.info(f"Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        
        logging.info(f"Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    
    # Save model weights
    os.makedirs(weight_dir, exist_ok=True)
    weight_path = os.path.join(weight_dir, f"{model_name}_fold_{fold}.pth")
    torch.save(model.state_dict(), weight_path)
    logging.info(f"Saved model weights to {weight_path}")

# Main script
if __name__ == "__main__":
    device = get_device()
    
    # Prepare the data paths and labels
    base_dir = 'data'
    log_dir = 'log'
    weight_dir = 'weights'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(filename=log_file, level=logging.INFO)
    
    patient_data = prepare_dataset(base_dir)

    # Set up transforms
    transform = get_transforms(train=True)

    # Choose model
    model_names = ['resnet152', 'densenet161', 'mobilenet']
    for model_name in model_names:
        # Five-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        criterion = nn.CrossEntropyLoss()

        fold = 1
        for train_index, val_index in kf.split(patient_data):
            logging.info(f"Fold {fold}")
            train_data = [patient_data[i] for i in train_index]
            val_data = [patient_data[i] for i in val_index]

            train_paths = [image_path for patient in train_data for image_path in patient[0]]
            train_labels = [patient[1] for patient in train_data for _ in patient[0]]
            val_paths = [image_path for patient in val_data for image_path in patient[0]]
            val_labels = [patient[1] for patient in val_data for _ in patient[0]]

            # Count class samples in the training set
            class_counts = Counter(train_labels)
            total_samples = len(train_labels)
            class_weights = {class_id: total_samples / count for class_id, count in class_counts.items()}
            weights = [class_weights[label] for label in train_labels]
            
            # Create weighted sampler for imbalanced datasets
            sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

            train_dataset = CAD_MENBI_Dataset(train_paths, train_labels, transform=transform)
            val_dataset = CAD_MENBI_Dataset(val_paths, val_labels, transform=transform)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

            # Reinitialize model for each fold
            model = select_model(model_name)

            # Define the optimizer
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Define the weighted loss function
            class_weights_tensor = torch.tensor([class_weights[i] for i in range(4)], dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss()
            
            # Train and validate the model
            train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, fold=fold, log_dir=log_dir, weight_dir=weight_dir, model_name=model_name)

            fold += 1
