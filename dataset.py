import os
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd


class MammoDataset(Dataset):
    def __init__(self,df,transform,image_dir,target='pathology'):
        self.df = df
        self.transform = transform
        self.target = target
        self.image_dir = image_dir
        self.label_map={'benign':0,'malignant':1}

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        image_name=self.df.iloc[index]['image file path']
        image_path=os.path.join(self.image_dir,image_name)
        image=Image.open(image_path).convert('RGB')
        label=self.label_map[self.df.iloc[index][self.target]]

        if self.transform is not None:
            image=self.transform(image)
        
        return image,label
    
def compute_mean_std(dataset):
    temp_loader=DataLoader(dataset,batch_size=16,shuffle=False)
    mean=0.
    std=0.
    total=0

    for images,_ in temp_loader:
        batch_samples=images.size(0)
        images=images.view(batch_samples,images.size(1),-1)
        mean+=images.mean(2).sum(0)
        std+=images.std(2).sum(0)
        total+=batch_samples    

    mean/=total
    std/=total  
    return mean.tolist(),std.tolist()

