import torch
import torch.optim as optim
from PIL import Image
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from dataset import MammoDataset,compute_mean_std
from models import MammoNet,Pretrained_ResNet
from config import *
import os
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def train_model(model,train_loader,val_loader,model_name,custom_criterion=None):
    device=torch.device('mps' if torch.mps.is_available() else 'cpu')
    model=model.to(device)
    criterion = custom_criterion if custom_criterion is not None else torch.nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=Learning_rate)
    
    best_accuracy=0
    for epochs in range(num_epochs):
        model.train()
        running_loss=0.0
        for images, labels in train_loader:
            images,labels=images.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()

        model.eval()
        correct=0   
        total=0
        with torch.no_grad():
            for images, labels in val_loader:
                images,labels=images.to(device),labels.to(device)
                outputs=model(images)
                _,predicted=torch.max(outputs.data,1)
                total+=labels.size(0)
                correct+=(predicted==labels).sum().item()

        accuracy=100*correct/total
        print(f'{model_name} Epoch {epochs+1}/{num_epochs} Loss: {running_loss/len(train_loader)} Accuracy: {accuracy}%')
        if accuracy>best_accuracy:
            best_accuracy=accuracy
            torch.save(model.state_dict(),f'{model_name}.pth')

    return best_accuracy

def main():
    calc_df = pd.read_csv(f'{CSV_dir}/calc_train_pro.csv')
    mass_df = pd.read_csv(f'{CSV_dir}/mass_train_pro.csv')
    combined_df = pd.concat([calc_df, mass_df], axis=0)
    
    print(f"DataFrame shape before cleaning: {combined_df.shape}")
    print(f"DataFrame 'image file path' column type: {combined_df['image file path'].dtype}")
    print(f"Number of NaN values in 'image file path': {combined_df['image file path'].isna().sum()}")
    
    combined_df = combined_df.dropna(subset=['image file path'])
    print(f"DataFrame shape after dropping NaNs: {combined_df.shape}")
    
    combined_df['image file path'] = combined_df['image file path'].astype(str)
    
    nan_str_mask = combined_df['image file path'].str.lower() == 'nan'
    if nan_str_mask.any():
        print(f"Found {nan_str_mask.sum()} 'nan' strings in paths, removing them")
        combined_df = combined_df[~nan_str_mask]
    
    valid_paths = []
    for path in combined_df['image file path']:
        try:
            full_path = os.path.join(Image_dir, path)
            if not os.path.exists(full_path):
                base, ext = os.path.splitext(full_path)
                alternatives = ['.jpg', '.jpeg', '']
                found = False
                for alt_ext in alternatives:
                    alt_path = base + alt_ext
                    if os.path.exists(alt_path):
                        valid_paths.append(path)
                        found = True
                        break
                if not found:
                    print(f"File not found for {full_path} (tried alternatives)")
            else:
                valid_paths.append(path)
        except Exception as e:
            print(f"Error processing path '{path}': {e}")
    
    print(f"Found {len(valid_paths)} valid paths out of {len(combined_df)}")
    combined_df = combined_df[combined_df['image file path'].isin(valid_paths)]
    
    train_df,val_df=train_test_split(combined_df,test_size=0.2,stratify=combined_df['pathology'])

    temporary_dataset=MammoDataset(train_df,transform=transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()],),image_dir=Image_dir)
    mean,std=compute_mean_std(temporary_dataset)

    DicomToPIL = transforms.Lambda(lambda x: x.convert('RGB') if isinstance(x,Image.Image) else x)

    global Train_transform_Mamonet,Val_transform_Mamonet,Train_transform_Resnet,Val_transform_Resnet
    Train_transform_Mamonet=transforms.Compose([DicomToPIL,
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
    Val_transform_Mamonet=transforms.Compose([DicomToPIL,transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
    Train_transform_Resnet = transforms.Compose([transforms.Grayscale(num_output_channels=3),
    transforms.RandomResizedCrop((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    Val_transform_Resnet = transforms.Compose([transforms.Grayscale(num_output_channels=3),
        transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    mamonet_train_dataset=MammoDataset(train_df,transform=Train_transform_Mamonet,image_dir=Image_dir)
    mamonet_val_dataset=MammoDataset(val_df,transform=Val_transform_Mamonet,image_dir=Image_dir)
    resnet_train_dataset=MammoDataset(train_df,transform=Train_transform_Resnet,image_dir=Image_dir)
    resnet_val_dataset=MammoDataset(val_df,transform=Val_transform_Resnet,image_dir=Image_dir)

    mamonnet_train_loader=DataLoader(mamonet_train_dataset,batch_size=batch_size,shuffle=True)
    mamonnet_val_loader=DataLoader(mamonet_val_dataset,batch_size=batch_size,shuffle=False) 
    resnet_train_loader=DataLoader(resnet_train_dataset,batch_size=batch_size,shuffle=True)
    resnet_val_loader=DataLoader(resnet_val_dataset,batch_size=batch_size,shuffle=False)

    MammoNet_model=MammoNet()
    Pretrained_ResNet_model=Pretrained_ResNet()

    print('Training MammoNet')
    # First, map pathology to actual numeric labels
    mapped_labels = train_df['pathology'].map({'BENIGN': 0, 'MALIGNANT': 1, 'BENIGN_WITHOUT_CALLBACK': 0}).values
    # Then compute weights on the binary labels
    class_weights = compute_class_weight('balanced', classes=np.unique(mapped_labels), y=mapped_labels)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device='mps')
    weighted_criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    MammoNet_accuracy = train_model(MammoNet_model, mamonnet_train_loader, mamonnet_val_loader, 'MammoNet', weighted_criterion)
    print('Training Pretrained ResNet')
    ResNet_accuracy = train_model(Pretrained_ResNet_model, resnet_train_loader, resnet_val_loader, 'Pretrained_ResNet', weighted_criterion)

    print('\nTraining Completed')
    print('\nResults:')
    print(f'MammoNet Accuracy: {MammoNet_accuracy} Pretrained ResNet Accuracy: {ResNet_accuracy}')

if __name__=='__main__':
    main()