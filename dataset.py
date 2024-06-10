import os
from PIL import Image
from torch.utils.data import Dataset

# Mapping of labels to numeric values
label_mapping = {"EGC": 0, "HGIN": 1, "LGIN": 2, "IM&CG": 3}

# Custom Dataset class
class CAD_MENBI_Dataset(Dataset):
    def __init__(self, data_paths, labels, transform=None):
        self.data_paths = data_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        image_path = self.data_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Function to prepare dataset
def prepare_dataset(base_dir):
    patient_data = []
    
    for label in os.listdir(base_dir):
        label_path = os.path.join(base_dir, label)
        if os.path.isdir(label_path):
            for year in os.listdir(label_path):
                year_path = os.path.join(label_path, year)
                if os.path.isdir(year_path):
                    for patient in os.listdir(year_path):
                        patient_path = os.path.join(year_path, patient)
                        if os.path.isdir(patient_path):
                            images = [os.path.join(patient_path, image) for image in os.listdir(patient_path)]
                            patient_data.append((images, label_mapping[label]))
    
    return patient_data
