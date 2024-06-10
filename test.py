import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torchvision import transforms

from dataset import CAD_MENBI_Dataset, prepare_dataset
from model import select_model
from utils import get_transforms

# Function to evaluate the model and compute metrics
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, model_name, save_path, average=False, font_size=18):
    plt.figure(figsize=(8, 6))
    if average:
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes, annot_kws={"size": font_size})
        plt.title(f'Average Confusion Matrix - {model_name}', fontsize=font_size)
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, annot_kws={"size": font_size})
        plt.title(f'Confusion Matrix - {model_name}', fontsize=font_size)
    plt.xlabel('Predicted Labels', fontsize=font_size)
    plt.ylabel('True Labels', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    
    plt.savefig(save_path)
    plt.close()

# Main script
if __name__ == "__main__":
    # Prepare the data paths and labels
    base_dir = 'data_selected'
    weight_dir = 'weights'
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)
    
    patient_data = prepare_dataset(base_dir)

    # Set up transforms
    transform = get_transforms(train=False)

    # Choose model
    model_names = ['resnet152', 'densenet161', 'mobilenet']
    classes = ["EGC", "HGIN", "LGIN", "IM&CG"]

    for model_name in model_names:
        model = select_model(model_name)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Five-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        all_cm = np.zeros((len(classes), len(classes)))

        fold = 1
        for train_index, val_index in kf.split(patient_data):
            # Load the model weights
            weight_path = os.path.join(weight_dir, f'{model_name}_fold_{fold}.pth')
            model.load_state_dict(torch.load(weight_path, map_location=device))
            
            # Prepare validation data
            val_data = [patient_data[i] for i in val_index]
            val_paths = [image_path for patient in val_data for image_path in patient[0]]
            val_labels = [patient[1] for patient in val_data for _ in patient[0]]
            
            val_dataset = CAD_MENBI_Dataset(val_paths, val_labels, transform=transform)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

            # Evaluate the model
            preds, labels = evaluate_model(model, val_loader, device)

            # Calculate metrics
            cm = confusion_matrix(labels, preds)
            all_cm += cm

            acc = accuracy_score(labels, preds)
            avg_recall = recall_score(labels, preds, average='macro')
            avg_precision = precision_score(labels, preds, average='macro')
            f1 = f1_score(labels, preds, average='macro')

            # Log metrics
            with open(os.path.join(save_dir, f'{model_name}.txt'), 'a') as f:
                f.write(f'Fold: {fold}\n')
                f.write(f'Accuracy: {acc:.4f}\n')
                f.write(f'Average Recall: {avg_recall:.4f}\n')
                f.write(f'Average Precision: {avg_precision:.4f}\n')
                f.write(f'F1 Score: {f1:.4f}\n')

            # Plot and save confusion matrix
            plot_confusion_matrix(cm, classes, model_name, os.path.join(save_dir, f'{model_name}_confusion_matrix_fold_{fold}.png'))

            fold += 1

        # Average the confusion matrix over the folds
        avg_cm = all_cm / fold
        avg_cm_percentage = avg_cm / avg_cm.sum(axis=1, keepdims=True) * 100

        # Save the average confusion matrix
        plot_confusion_matrix(avg_cm_percentage, classes, model_name, os.path.join(save_dir, f'{model_name}_average_confusion_matrix.png'), average=True)
