import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import math
import pandas as pd

class imageDataset(Dataset):
    def __init__(self, root_dir, batch_size: int, total_num_samp: int, target_data, filenames, randomize: bool):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.total_num_samp = total_num_samp
        self.target_data = target_data
        self.filenames = filenames
        self.randomize = randomize

        self.iter = None
        self.num_batches_per_epoch = math.ceil(total_num_samp / self.batch_size)
        self.image_list = os.listdir(root_dir)
        

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, filename):
        image_path = os.path.join(self.root_dir, filename)
        
        # Load image and apply preprocessing
        image = Image.open(image_path).convert('L') # <----grey scale (This line pulls directly from disk)
        image = image.resize((200, 200)) # <----ensure that each image is the same size
        
        transform = transforms.Compose([
            transforms.ToTensor(), # <---- Coverts PIL image to torch tensor
            transforms.Normalize(mean=[0.5], std=[0.5])  # <---- Adjust mean and std as needed for standardization
        ])

        image = transform(image)
        
        return image
    
    def randomize_dataset(self):
        indices = torch.randperm(self.total_num_samp)
        self.target_data = [self.target_data[i] for i in indices]
        self.filenames = [self.filenames[i] for i in indices] # <---- Have to itterate through since its not a tensor
        
        
    def generate_iterable(self):
        """
        This function converts the dataset into a sequence of batches, and wraps it in
        an iterable that can be called to efficiently fetch one batch at a time
        """
    
        if self.randomize:
            self.randomize_dataset()

        # split dataset into sequence of batches 
        batches = []
        for b_idx in range(self.num_batches_per_epoch):
            batches.append(
                {
                'filename_batch':self.filenames[b_idx * self.batch_size : (b_idx+1) * self.batch_size],
                'target_batch':self.target_data[b_idx * self.batch_size : (b_idx+1) * self.batch_size],
                'batch_idx':b_idx,
                }
            )
        self.iter = iter(batches)
    
    def generate_batch(self):
        if self.iter == None:
            self.generate_iterable()

        batch = next(self.iter)


        # For-loop to grab corresponsing images based on filename
        image_batch = []
        for filename in batch['filename_batch']:
            x = self.__getitem__(filename)
            image_batch.append(x)


        if batch['batch_idx'] == self.num_batches_per_epoch - 1:
            # generate a fresh iterable
            self.generate_iterable()

        return torch.stack(image_batch), torch.tensor(batch['target_batch']).float() 


class MultimodalDataset(Dataset):
    def __init__(self, root_dir, batch_size: int, total_num_samp: int, target_data, feature_data, filenames, randomize: bool):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.total_num_samp = total_num_samp
        self.target_data = target_data
        self.feature_data = feature_data
        self.filenames = filenames
        self.randomize = randomize

        self.iter = None
        self.num_batches_per_epoch = math.ceil(total_num_samp / self.batch_size)
        self.image_list = os.listdir(root_dir)
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, filename):
        image_path = os.path.join(self.root_dir, filename)
        
        # Load image and apply preprocessing
        image = Image.open(image_path).convert('L') # <----grey scale (This line pulls directly from disk)
        image = image.resize((200, 200)) # <----ensure that each image is the same size
        
        transform = transforms.Compose([
            transforms.ToTensor(), # <---- Coverts PIL image to torch tensor
            transforms.Normalize(mean=[0.5], std=[0.5])  # <---- Adjust mean and std as needed for standardization
        ])

        image = transform(image)
        
        return image
    
    def randomize_dataset(self):
        indices = torch.randperm(self.total_num_samp)
        self.target_data = [self.target_data[i] for i in indices]
        self.filenames = [self.filenames[i] for i in indices]

    def generate_iterable(self):
    
        if self.randomize:
            self.randomize_dataset()

        # split dataset into sequence of batches 
        batches = []
        for b_idx in range(self.num_batches_per_epoch):
            batches.append(
                {
                'filename_batch':self.filenames[b_idx * self.batch_size : (b_idx+1) * self.batch_size],
                'target_batch':self.target_data[b_idx * self.batch_size : (b_idx+1) * self.batch_size],
                'feature_batch':self.feature_data[b_idx * self.batch_size : (b_idx+1) * self.batch_size],
                'batch_idx':b_idx,
                }
            )
        self.iter = iter(batches)
    
    def generate_batch(self):
        if self.iter == None:
            self.generate_iterable()

        batch = next(self.iter)

        # For-loop to grab corresponsing images based on filename
        image_batch = []
        for filename in batch['filename_batch']:
            x = self.__getitem__(filename)
            image_batch.append(x)


        if batch['batch_idx'] == self.num_batches_per_epoch - 1:
            # generate a fresh iterable
            self.generate_iterable()


        return torch.stack(image_batch), torch.tensor(batch['target_batch']).float(), torch.tensor(batch['feature_batch']).float()
    



