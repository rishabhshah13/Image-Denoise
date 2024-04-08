import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        # Iterate through folders and collect image paths
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                for image_name in os.listdir(os.path.join(folder_path)):
                    self.image_paths.append({
                        'gt': os.path.join(folder_path, image_name),
                        'noisy': os.path.join(folder_path, image_name)
                    })

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        gt_image_path = self.image_paths[idx]['gt']
        noisy_image_path = self.image_paths[idx]['noisy']
        
        gt_image = Image.open(gt_image_path).convert('RGB')
        noisy_image = Image.open(noisy_image_path).convert('RGB')

        if self.transform:
            gt_image = self.transform(gt_image)
            noisy_image = self.transform(noisy_image)

        return gt_image, noisy_image

# # Define transformations to be applied to the images
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),  # Resize the image
#     transforms.ToTensor(),           # Convert to tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
# ])

# # Set root directory
# # root_dir = 'data'
# root_dir='SIDD_Small_sRGB_Only/Data/' #dataset directory


# # Create dataset
# dataset = CustomDataset(root_dir, transform=transform)

# # Split dataset into train and test sets
# train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# # Create dataloaders for train and test sets
# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
