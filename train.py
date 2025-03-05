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

def train_model(model,train_loader,val_loader,model_name):
    device=torch.device('mps' if torch.mps.is_available() else 'cpu')
    model=model.to(device)
    criterion=torch.nn.CrossEntropyLoss()
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
    calc_df=pd.read_csv(f'{CSV_dir}/calc_case_description_train_set.csv')
    mass_df=pd.read_csv(f'{CSV_dir}/mass_case_description_train_set.csv')
    combined_df=pd.concat([calc_df,mass_df])
    train_df,val_df=train_test_split(combined_df,test_size=0.2,stratify=combined_df['pathology'])

    temporary_dataset=MammoDataset(train_df,transform=None,image_dir=Image_dir)
    mean,std=compute_mean_std(temporary_dataset)

    DicomToPIL = transforms.Lambda(lambda x: x.convert('RGB') if isinstance(x,Image.Image) else x)

    global Train_transform_Mamonet,Val_transform_Mamonet,Train_transform_Resnet,Val_transform_Resnet
    Train_transform_Mamonet=transforms.Compose([DicomToPIL,
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
    Val_transform_Mamonet=transforms.Compose([DicomToPIL,
        transforms.Resize(256),transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
    Train_transform_Resnet = transforms.Compose([transforms.Grayscale(num_output_channels=3),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    Val_transform_Resnet = transforms.Compose([transforms.Grayscale(num_output_channels=3),
        transforms.Resize(256),
        transforms.CenterCrop(224),
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
    MammoNet_accuracy=train_model(MammoNet_model,mamonnet_train_loader,mamonnet_val_loader,'MammoNet')
    print('Training Pretrained ResNet')
    ResNet_accuracy=train_model(Pretrained_ResNet_model,resnet_train_loader,resnet_val_loader,'Pretrained_ResNet')

    print('\nTraining Completed')
    print('\nResults:')
    print(f'MammoNet Accuracy: {MammoNet_accuracy} Pretrained ResNet Accuracy: {ResNet_accuracy}')

if __name__=='__main__':
    main()