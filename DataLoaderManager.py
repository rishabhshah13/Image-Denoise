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





import matplotlib.pyplot as plt  # Import for using plt
def display_progress(cond, real, fake, figsize=(10,5)):
    cond = cond.detach().cpu().permute(1, 2, 0)
    fake = fake.detach().cpu().permute(1, 2, 0)
    real = real.detach().cpu().permute(1, 2, 0)

    fig, ax = plt.subplots(1, 3, figsize=figsize)
    ax[0].imshow(cond)
    ax[1].imshow(real)
    ax[2].imshow(fake)
    plt.show()




class DataLoaderManager:
    def __init__(self, root_dir, train_file,test_file,make_held_out_set=False,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        # self.train_file = f'Dataset\{train_file}_train_dataloader.npy'
        # self.test_file = f'Dataset\{test_file}_train_dataloader.npy'
        self.train_file = f'{train_file}_train_dataloader.npy'
        self.test_file = f'{test_file}_test_dataloader.npy'
        self.make_held_out_set  = make_held_out_set

        folder_name = "Datasets"

        # Check if the folder exists
        if not os.path.exists(folder_name):
            # If it doesn't exist, create it
            os.makedirs(folder_name)
            print(f"Folder '{folder_name}' created.")
        else:
            print(f"Folder '{folder_name}' already exists.")


        print("Collecting image paths...")

        if self.make_held_out_set == True:
            folder_list = ['0135_006_IP_00400_00400_5500_N', '0138_006_IP_00100_00100_3200_L', '0180_008_GP_00100_00100_5500_N', '0070_003_IP_02000_04000_3200_N', '0121_006_N6_03200_01000_3200_L', '0035_002_GP_00800_00350_3200_N', '0130_006_GP_00400_00400_4400_N', '0065_003_GP_10000_08460_4400_N', '0129_006_GP_00100_00100_4400_N', '0181_008_GP_00800_00800_5500_N', '0022_001_N6_00100_00060_5500_N', '0108_005_GP_06400_06400_4400_N', '0017_001_GP_00100_00060_5500_N', '0086_004_GP_00100_00100_5500_L', '0029_001_IP_00800_01000_5500_N', '0036_002_GP_06400_03200_3200_N', '0164_007_IP_00400_00400_3200_N', '0188_008_IP_00100_00100_3200_N', '0126_006_S6_00400_00200_4400_L', '0018_001_GP_00100_00160_5500_L', '0097_005_N6_03200_02000_3200_L', '0020_001_GP_00800_00350_5500_N', '0014_001_S6_03200_01250_3200_N', '0011_001_S6_00800_00500_5500_L', '0038_002_GP_00800_00640_3200_L', '0192_009_IP_00100_00200_3200_N', '0125_006_S6_00100_00050_4400_L', '0072_003_IP_01000_02000_5500_L', '0167_008_N6_00100_00050_4400_L', '0134_006_IP_00100_00100_5500_N', '0173_008_G4_00400_00400_4400_N', '0101_005_S6_00100_00050_4400_L']


        ## linkedin Link
        ## Primary Contact

        else:
            print("NOT HELDOUT")
            folder_list = os.listdir(root_dir)
            heldout_list = ['0135_006_IP_00400_00400_5500_N', '0138_006_IP_00100_00100_3200_L', '0180_008_GP_00100_00100_5500_N', '0070_003_IP_02000_04000_3200_N', '0121_006_N6_03200_01000_3200_L', '0035_002_GP_00800_00350_3200_N', '0130_006_GP_00400_00400_4400_N', '0065_003_GP_10000_08460_4400_N', '0129_006_GP_00100_00100_4400_N', '0181_008_GP_00800_00800_5500_N', '0022_001_N6_00100_00060_5500_N', '0108_005_GP_06400_06400_4400_N', '0017_001_GP_00100_00060_5500_N', '0086_004_GP_00100_00100_5500_L', '0029_001_IP_00800_01000_5500_N', '0036_002_GP_06400_03200_3200_N', '0164_007_IP_00400_00400_3200_N', '0188_008_IP_00100_00100_3200_N', '0126_006_S6_00400_00200_4400_L', '0018_001_GP_00100_00160_5500_L', '0097_005_N6_03200_02000_3200_L', '0020_001_GP_00800_00350_5500_N', '0014_001_S6_03200_01250_3200_N', '0011_001_S6_00800_00500_5500_L', '0038_002_GP_00800_00640_3200_L', '0192_009_IP_00100_00200_3200_N', '0125_006_S6_00100_00050_4400_L', '0072_003_IP_01000_02000_5500_L', '0167_008_N6_00100_00050_4400_L', '0134_006_IP_00100_00100_5500_N', '0173_008_G4_00400_00400_4400_N', '0101_005_S6_00100_00050_4400_L']
            new_list = [img_name for img_name in folder_list if img_name not in heldout_list]
            folder_list = new_list
        
        # for folder_name in tqdm(folder_list):
        #     folder_path = os.path.join(root_dir, folder_name)
        #     # print(folder_path)
        #     if os.path.isdir(folder_path):
        #         image_names = os.listdir(os.path.join(folder_path))
        #         self.image_paths.append({
        #                 'gt': os.path.join(folder_path, image_names[0]),
        #                 'noisy': os.path.join(folder_path, image_names[1])
        #             })

        import glob
        for folder_name in tqdm(folder_list):
            folder_path = os.path.join(root_dir, folder_name)
            # print(folder_path)
            
            if os.path.isdir(folder_path):
                # image_names = os.listdir(os.path.join(folder_path))
                GT_images = glob.glob(os.path.join(folder_path) + '/*GT*')
                NOISY_images = glob.glob(os.path.join(folder_path) + '/*NOISY*')
                for i in range(len(GT_images)):
                    self.image_paths.append({
                            'gt': GT_images[i],
                            'noisy': NOISY_images[i]
                            })
            

    def create_datasets(self):
        print("Creating datasets...")
        dataset = CustomDataset(self.image_paths, transform=self.transform)
        
        print("Splitting dataset into train and test sets...")
        if self.make_held_out_set == False:
            train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
            return train_dataset, test_dataset
        else:
            return dataset, []

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
        data_list = []
        import torch
        for i in range(data.shape[0]):
            data_tuple = (torch.tensor(data[i][0]), torch.tensor(data[i][1]))
            data_list.append(data_tuple)

        return DataLoader(data_list, batch_size=batch_size, shuffle=False,num_workers=32, persistent_workers=True,pin_memory=True)
    

    def process_dataloaders(self, batch_size=32, shuffle=True):
        if os.path.exists(self.train_file) and os.path.exists(self.test_file):
            train_dataloader = self.load_dataloader_from_npy(self.train_file, batch_size, False, is_train=True)
            test_dataloader = self.load_dataloader_from_npy(self.test_file, batch_size, False, is_train=False)
            print("DataLoader loaded from .npy files.")
        else:
            train_dataset, test_dataset = self.create_datasets()
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=32, persistent_workers=True,pin_memory=True)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=32, persistent_workers=True,pin_memory=True)
            
            
            if self.make_held_out_set == True:
                print(f"Creating {self.train_file}")
                self.save_dataloader_to_npy(train_dataloader, self.train_file)
            else:
                print(f"Creating {self.train_file}")
                print(f"Creating {self.test_file}")
                self.save_dataloader_to_npy(train_dataloader, self.train_file)
                self.save_dataloader_to_npy(test_dataloader, self.test_file)

        return train_dataloader, test_dataloader