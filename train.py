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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, confusion_matrix

def train_model(model,train_loader,val_loader,model_name,custom_criterion=None,patience=7):
    device=torch.device('mps' if torch.mps.is_available() else 'cpu')
    model=model.to(device)
    criterion = custom_criterion if custom_criterion is not None else torch.nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=Learning_rate,weight_decay=weight_decay)

    scheduler=ReduceLROnPlateau(optimizer,mode='max',factor=0.5,patience=3,verbose=True)
    patience_counter=0
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
        
        #udating the learning rate using validation performance
        scheduler.step(accuracy)

        if accuracy>best_accuracy:
            best_accuracy=accuracy
            torch.save(model.state_dict(),f'{model_name}.pth')
            patience_counter=0
        else:
            patience_counter+=1
            if patience_counter>=patience:
                print(f"Early stopping at epoch {epochs+1}")
                break

    return best_accuracy

def main():
    device=torch.device('mps' if torch.mps.is_available() else 'cpu')
    calc_df = pd.read_csv(f'{CSV_dir}/calc_train_pro.csv')
    mass_df = pd.read_csv(f'{CSV_dir}/mass_train_pro.csv')
    combined_df = pd.concat([calc_df, mass_df], axis=0)
    calc_test_df = pd.read_csv(f'{CSV_dir}/calc_test_pro.csv')
    mass_test_df = pd.read_csv(f'{CSV_dir}/mass_test_pro.csv')
    test_df = pd.concat([calc_test_df, mass_test_df], axis=0)

    del calc_df,mass_df, calc_test_df, mass_test_df
    
    print(f"DataFrame shape before cleaning: {combined_df.shape}")
    print(f"DataFrame 'image file path' column type: {combined_df['image file path'].dtype}")
    print(f"Number of NaN values in 'image file path': {combined_df['image file path'].isna().sum()}")
    
    combined_df = combined_df.dropna(subset=['image file path'])
    test_df=test_df.dropna(subset=['image file path'])
    test_df['image file path']=test_df['image file path'].astype(str)
    
    print(f"DataFrame shape after dropping NaNs: {combined_df.shape}")
    
    combined_df['image file path'] = combined_df['image file path'].astype(str)
    
    nan_str_mask = combined_df['image file path'].str.lower() == 'nan'
    if nan_str_mask.any():
        print(f"Found {nan_str_mask.sum()} 'nan' strings in paths, removing them")
        combined_df = combined_df[~nan_str_mask]
    
    train_df,val_df=train_test_split(combined_df,test_size=0.2,stratify=combined_df['pathology'])

    temporary_dataset=MammoDataset(train_df,transform=transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()],),image_dir=Image_dir)
    mean,std=compute_mean_std(temporary_dataset)

    DicomToPIL = transforms.Lambda(lambda x: x.convert('RGB') if isinstance(x,Image.Image) else x)

    global Train_transform_Mamonet,Val_transform_Mamonet,Train_transform_Resnet,Val_transform_Resnet
    Train_transform_Mamonet=transforms.Compose([DicomToPIL,
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
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
    transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),  
    transforms.RandomRotation(15),    
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # brightness/contrast variation
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
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
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    weighted_criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    MammoNet_accuracy = train_model(MammoNet_model, mamonnet_train_loader, mamonnet_val_loader, 'MammoNet', weighted_criterion)
    print('Training Pretrained ResNet')
    ResNet_accuracy = train_model(Pretrained_ResNet_model, resnet_train_loader, resnet_val_loader, 'Pretrained_ResNet', weighted_criterion)

    print('\nTraining Completed')
    print('\nResults:')
    print(f'MammoNet Accuracy: {MammoNet_accuracy} Pretrained ResNet Accuracy: {ResNet_accuracy}')

    #test data
    mamonet_test_dataset = MammoDataset(test_df, transform=Val_transform_Mamonet, image_dir=Image_dir)
    resnet_test_dataset = MammoDataset(test_df, transform=Val_transform_Resnet, image_dir=Image_dir)
    
    mamonet_test_loader = DataLoader(mamonet_test_dataset, batch_size=batch_size, shuffle=False)
    resnet_test_loader = DataLoader(resnet_test_dataset, batch_size=batch_size, shuffle=False)
    
  
    print("\nEvaluating models on test set...")
    
    # Load best models
    MammoNet_model.load_state_dict(torch.load('MammoNet.pth'))
    Pretrained_ResNet_model.load_state_dict(torch.load('Pretrained_ResNet.pth'))
    
    #evaluating mammonet
    MammoNet_model.eval()
    mamonet_correct = 0
    mamonet_total = 0
    all_mamonet_labels = []
    all_mamonet_probs = []
    all_mamonet_preds = []

    with torch.no_grad():
        for images, labels in mamonet_test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = MammoNet_model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            mamonet_total += labels.size(0)
            mamonet_correct += (predicted == labels).sum().item()
            all_mamonet_labels.extend(labels.cpu().numpy())
            all_mamonet_probs.extend(probs[:, 1].cpu().numpy())
            all_mamonet_preds.extend(predicted.cpu().numpy())

    mamonet_test_acc = 100 * mamonet_correct / mamonet_total
    print(f"MammoNet Test Accuracy: {mamonet_test_acc:.2f}%")

    # MammoNet metrics
    auc_mamonet = roc_auc_score(all_mamonet_labels, all_mamonet_probs)
    tn, fp, fn, tp = confusion_matrix(all_mamonet_labels, all_mamonet_preds).ravel()
    sensitivity_mamonet = tp / (tp + fn)
    specificity_mamonet = tn / (tn + fp)

    print(f"MammoNet Metrics - Sensitivity: {sensitivity_mamonet:.4f}, Specificity: {specificity_mamonet:.4f}, AUC: {auc_mamonet:.4f}")

    # For ResNet50
    Pretrained_ResNet_model.eval()
    resnet_correct = 0
    resnet_total = 0
    all_resnet_labels = []
    all_resnet_probs = []
    all_resnet_preds = []

    with torch.no_grad():
        for images, labels in resnet_test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = Pretrained_ResNet_model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            resnet_total += labels.size(0)
            resnet_correct += (predicted == labels).sum().item()
            all_resnet_labels.extend(labels.cpu().numpy())
            all_resnet_probs.extend(probs[:, 1].cpu().numpy())
            all_resnet_preds.extend(predicted.cpu().numpy())

    resnet_test_acc = 100 * resnet_correct / resnet_total
    print(f"ResNet50 Test Accuracy: {resnet_test_acc:.2f}%")

    # ResNet metrics
    auc_resnet = roc_auc_score(all_resnet_labels, all_resnet_probs)
    tn, fp, fn, tp = confusion_matrix(all_resnet_labels, all_resnet_preds).ravel()
    sensitivity_resnet = tp / (tp + fn)
    specificity_resnet = tn / (tn + fp)

    print(f"ResNet50 Metrics - Sensitivity: {sensitivity_resnet:.4f}, Specificity: {specificity_resnet:.4f}, AUC: {auc_resnet:.4f}")

    # Final comparison table
    print("\n======= Model Performance Comparison =======")
    print(f"{'Model':<20} {'Val Acc':<10} {'Test Acc':<10} {'Sensitivity':<12} {'Specificity':<12} {'AUC':<10}")
    print(f"{'-'*64}")
    print(f"{'MammoNet':<20} {MammoNet_accuracy:<10.2f} {mamonet_test_acc:<10.2f} {sensitivity_mamonet:<12.4f} {specificity_mamonet:<12.4f} {auc_mamonet:<10.4f}")
    print(f"{'ResNet50':<20} {ResNet_accuracy:<10.2f} {resnet_test_acc:<10.2f} {sensitivity_resnet:<12.4f} {specificity_resnet:<12.4f} {auc_resnet:<10.4f}")

if __name__=='__main__':
    main()
