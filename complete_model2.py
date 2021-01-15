

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix


class WESADDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.drop('subject', axis=1)
        self.labels = self.dataframe['label'].values
        self.dataframe.drop('label', axis=1, inplace=True)
        
    def __getitem__(self, idx):
        x = self.dataframe.iloc[idx].values
        y = self.labels[idx]
        return torch.Tensor(x), y

    def __len__(self):
        return len(self.dataframe)


# ### Cross Validation Loader

# In[2]:


feats =   ['BVP_mean', 'BVP_std', 'BVP_min', 'BVP_max',
           'EDA_phasic_mean', 'EDA_phasic_std', 'EDA_phasic_min', 'EDA_phasic_max', 'EDA_smna_mean',
           'EDA_smna_std', 'EDA_smna_min', 'EDA_smna_max', 'EDA_tonic_mean',
           'EDA_tonic_std', 'EDA_tonic_min', 'EDA_tonic_max', 'Resp_mean',
           'Resp_std', 'Resp_min', 'Resp_max', 'TEMP_mean', 'TEMP_std', 'TEMP_min',
           'TEMP_max', 'TEMP_slope', 'BVP_peak_freq', 'age', 'height',
           'weight','subject', 'label']
layer_1_dim = len(feats) -2
print(layer_1_dim)


# In[14]:


def get_data_loaders(df, subject_id, train_batch_size=25, test_batch_size=5):
    #df = pd.read_csv('data/m14_merged.csv', index_col=0)[feats]

    train_df = df[ df['subject'] != subject_id].reset_index(drop=True)
    test_df = df[ df['subject'] == subject_id].reset_index(drop=True)
    
    train_dset = WESADDataset(train_df)
    test_dset = WESADDataset(test_df)

    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=test_batch_size)
    
    return train_loader, test_loader


# ## Network Architecture

# In[16]:


class StressNet(nn.Module):
    def __init__(self):
        super(StressNet, self).__init__()
        self.fc = nn.Sequential(
                        nn.Linear(29, 128),
                        #nn.Dropout(0.5),
                        nn.ReLU(),
                        nn.Linear(128, 256),
                        #nn.Dropout(0.5),
                        nn.ReLU(),
                        nn.Linear(256, 2),
                        #nn.Dropout(0.5),
                        nn.LogSoftmax(dim=1))
        
    def forward(self, x):
        return self.fc(x)    


# ## Model Training

# In[5]:


def train(model, optimizer, train_loader, validation_loader):
    history = {'train_loss': {}, 'train_acc': {}, 'valid_loss': {}, 'valid_acc': {}}
    #
    for epoch in range(num_epochs):

        # Train:   
        total = 0
        correct = 0
        trainlosses = []

        for batch_index, (images, labels) in enumerate(train_loader):

            # Send to GPU (device)
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images.float())

            # Loss
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trainlosses.append(loss.item())

            # Compute accuracy
            _, argmax = torch.max(outputs, 1)
            correct += (labels == argmax).sum().item() #.mean()
            total += len(labels)

        history['train_loss'][epoch] = np.mean(trainlosses) 
        history['train_acc'][epoch] = correct/total 

        if epoch % 10 == 0:
            with torch.no_grad():

                losses = []
                total = 0
                correct = 0

                for images, labels in validation_loader:
                    # 
                    images, labels = images.to(device), labels.to(device)

                    # Forward pass
                    outputs = model(images.float())
                    loss = criterion(outputs, labels)

                    # Compute accuracy
                    _, argmax = torch.max(outputs, 1)
                    correct += (labels == argmax).sum().item() #.mean()
                    total += len(labels)

                    losses.append(loss.item())
                    
                history['valid_acc'][epoch] = np.round(correct/total, 3)
                history['valid_loss'][epoch] = np.mean(losses)

                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {np.mean(losses):.4}, Acc: {correct/total:.2}')
                
    return history


# In[6]:


def test(model, validation_loader):
    print('Evaluating model...')
    # Test
    model.eval()

    total = 0
    correct = 0
    testlosses = []
    correct_labels = []
    predictions = []

    with torch.no_grad():

        for batch_index, (images, labels) in enumerate(validation_loader):
            # Send to GPU (device)
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images.float())

            # Loss
            loss = criterion(outputs, labels)

            testlosses.append(loss.item())

            # Compute accuracy
            _, argmax = torch.max(outputs, 1)
            correct += (labels == argmax).sum().item() #.mean()
            total += len(labels)

            correct_labels.extend(labels)
            predictions.extend(argmax)


    test_loss = np.mean(testlosses)
    accuracy = np.round(correct/total, 2)
    print(f'Loss: {test_loss:.4}, Acc: {accuracy:.2}')
    
    y_true = [label.item() for label in correct_labels]
    y_pred = [label.item() for label in predictions]

    cm = confusion_matrix(y_true, y_pred)
    # TODO: return y true and y pred, make cm after ( use ytrue/ypred for classification report)
    # return [y_true, y_pred, test_loss, accuracy]
    return cm, test_loss, accuracy


# ## Start Here

# In[11]:


df = pd.read_csv('data/m14_merged.csv', index_col=0)
subject_id_list = df['subject'].unique()
df.head()


# In[12]:


def change_label(label):
    if label == 0 or label == 1:
        return 0
    else:
        return 1
    
df['label'] = df['label'].apply(change_label)


# In[18]:


df = df[feats]


# In[19]:


# Batch sizes
train_batch_size = 25
test_batch_size = 5

# Learning Rate
learning_rate = 5e-3

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Number of Epochs
num_epochs = 100

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

# models = [] # save models at all/ directly?
histories = []
confusion_matrices = []
test_losses = []
test_accs = []

for _ in subject_id_list:
    print('\nSubject: ', _)
    model = StressNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_loader, test_loader = get_data_loaders(df, _)
    
    history = train(model, optimizer, train_loader, test_loader)
    histories.append(history)
    
    cm, test_loss, test_acc = test(model, test_loader)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    confusion_matrices.append(cm)


# In[20]:


np.mean(test_accs)


# In[21]:


np.mean(test_losses)


# In[25]:


df['label'].value_counts()


# In[22]:


plt.figure(figsize=(14, 6))
plt.title('Testing Accuracies in Leave One Out Cross Validation by Subject Left Out as Testing Data')
sns.barplot(x=subject_id_list, y=test_accs);


# In[23]:


plt.figure(figsize=(14, 3))
plt.title('Testing Losses in Leave One Out Cross Validation by Subject Left Out as Testing Data')
sns.barplot(x=subject_id_list, y=test_losses);


# In[16]:


#infodf = pd.read_csv('data/WESAD/readmes.csv', index_col=0)
#infodf.sort_index()


# ## Training Visualization

# In[14]:


len(histories)


# ## Model Evaluation

# In[26]:


plt.figure(figsize=(15,10))

for i in range(15):
    plt.subplot(4,5 ,i+1)
    cm = confusion_matrices[i]
    
    
    sns.heatmap(cm, annot=True, fmt='d', cbar=False);
    plt.title(f'S{subject_id_list[i]}')
    plt.xlabel('Prediction');
    plt.ylabel('Ground Truth');
plt.tight_layout();


# In[16]:


#from sklearn.metrics import classification_report

#target_names = ['Amusement', 'Baseline', 'Stress']
#print(classification_report(y_true, y_pred, target_names=target_names))


# In[17]:


#torch.save(model.state_dict(), 'm13_model.pt')


# In[ ]:




