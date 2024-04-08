import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        gt_image_path = self.data[idx]['gt']
        noisy_image_path = self.data[idx]['noisy']
        
        gt_image = Image.open(gt_image_path).convert('RGB')
        noisy_image = Image.open(noisy_image_path).convert('RGB')

        if self.transform:
            gt_image = self.transform(gt_image)
            noisy_image = self.transform(noisy_image)

        return gt_image, noisy_image

class DataLoaderManager:
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        print("Collecting image paths...")
        for folder_name in tqdm(os.listdir(root_dir)[:10]):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                for image_name in os.listdir(os.path.join(folder_path)):
                    self.image_paths.append({
                        'gt': os.path.join(folder_path, image_name),
                        'noisy': os.path.join(folder_path, image_name)
                    })

    def create_datasets(self):
        print("Creating datasets...")
        dataset = CustomDataset(self.image_paths, transform=self.transform)
        
        print("Splitting dataset into train and test sets...")
        train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
        
        return train_dataset, test_dataset

    def save_dataloader_to_npy(self, dataloader, filename):
        print("Saving DataLoader to .npy file...")
        data = []
        labels = []

        for batch in tqdm(dataloader):
            gt_images, noisy_images = batch
            for gt_image, noisy_image in zip(gt_images, noisy_images):
                data.append((gt_image.numpy(), noisy_image.numpy()))

        np.save(filename, data)


    def load_dataloader_from_npy(self, filename, batch_size, shuffle, is_train):
        print("Loading DataLoader from .npy file...")
        data = np.load(filename, allow_pickle=True)
        # if is_train:
        #     data = data[:len(data)//2]  # Selecting train data
        # else:
        #     data = data[len(data)//2:]  # Selecting test data
        return DataLoader(data, batch_size=batch_size, shuffle=shuffle)

    def process_dataloaders(self, batch_size=32, shuffle=True):
        train_filename = 'train_dataloader.npy'
        test_filename = 'test_dataloader.npy'

        if os.path.exists(train_filename) and os.path.exists(test_filename):
            train_dataloader = self.load_dataloader_from_npy(train_filename, batch_size, shuffle, is_train=True)
            test_dataloader = self.load_dataloader_from_npy(test_filename, batch_size, False, is_train=False)
            print("DataLoader loaded from .npy files.")
        else:
            train_dataset, test_dataset = self.create_datasets()
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            self.save_dataloader_to_npy(train_dataloader, train_filename)
            self.save_dataloader_to_npy(test_dataloader, test_filename)

        return train_dataloader, test_dataloader
    

    # def save_dataloader_to_npy(self, dataloader, filename):
    #     print("Saving DataLoader to .npy file...")
    #     data = []
    #     import numpy as np
        
    #     for batch in tqdm(dataloader):
    #         gt_images, noisy_images = batch
    #         for gt_image, noisy_image in zip(gt_images, noisy_images):
    #             data.append((gt_image.numpy(), noisy_image.numpy()))
        
    #     np.save(filename, data)


    # def load_dataloader_from_npy(self, filename, batch_size, shuffle):
    #     print("Loading DataLoader from .npy file...")
    #     data = np.load(filename, allow_pickle=True)
    #     # dataset = CustomDataset(data, transform=self.transform)
    #     return DataLoader(data, batch_size=batch_size, shuffle=shuffle)

    # def process_dataloaders(self, batch_size=32, shuffle=True):
    #     train_filename = 'train_dataloader.npy'
    #     test_filename = 'test_dataloader.npy'

    #     if os.path.exists(train_filename) and os.path.exists(test_filename):
    #         train_dataloader = self.load_dataloader_from_npy(train_filename, batch_size, shuffle)
    #         test_dataloader = self.load_dataloader_from_npy(test_filename, batch_size, False)
    #         print("DataLoader loaded from .npy files.")
    #     else:
    #         train_dataset, test_dataset = self.create_datasets()
    #         train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    #         test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    #         self.save_dataloader_to_npy(train_dataloader, train_filename)
    #         self.save_dataloader_to_npy(test_dataloader, test_filename)

    #     return train_dataloader, test_dataloader

# # Example usage:
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),  # Resize the image
#     transforms.ToTensor(),           # Convert to tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
# ])

# data_loader_manager = DataLoaderManager(root_dir='SIDD_Small_sRGB_Only/Data/', transform=transform)
# train_dataloader, test_dataloader = data_loader_manager.process_dataloaders(batch_size=32, shuffle=True)
