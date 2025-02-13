# import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import numpy as np, pandas as pd
import matplotlib.pyplot as plt

# load data
file_path1 = "./stiffness_debug_a.csv"
file_path2 = "./stiffness_debug_b.csv"

idx0 = [1, 1211, 2421, 3631, 4841]
idx10 = [6051, 7261, 8471, 9681, 10891]
idx20 = [12101, 13311, 14521, 15731, 16941]
idx30 = [18151, 19361, 20571, 21781, 22991]

curve0, curve10, curve20, curve30 = [], [], [], []

for i in np.arange(0, 5):
    curve0.append(pd.read_csv(file_path1, skiprows=idx0[i], nrows=1200).values[:, [0,3]])
    curve0.append(pd.read_csv(file_path2, skiprows=idx0[i], nrows=1200).values[:, [0,3]])
    
    curve10.append(pd.read_csv(file_path1, skiprows=idx10[i], nrows=1200).values[:, [0,3]])
    curve10.append(pd.read_csv(file_path2, skiprows=idx10[i], nrows=1200).values[:, [0,3]])
    
    curve20.append(pd.read_csv(file_path1, skiprows=idx20[i], nrows=1200).values[:, [0,3]])
    curve20.append(pd.read_csv(file_path2, skiprows=idx20[i], nrows=1200).values[:, [0,3]])
    
    curve30.append(pd.read_csv(file_path1, skiprows=idx30[i], nrows=1200).values[:, [0,3]])
    curve30.append(pd.read_csv(file_path2, skiprows=idx30[i], nrows=1200).values[:, [0,3]])

curve0 = np.array(curve0).reshape(-1, 2)
curve10 = np.array(curve10).reshape(-1, 2)
curve20 = np.array(curve20).reshape(-1, 2)
curve30 = np.array(curve30).reshape(-1, 2)

slot = 30
slot_num = int(len(curve0)/slot)
num_classes = 4  

curves = np.concatenate((curve0, curve10, curve20, curve30), axis=0)
data = curves.reshape(-1, slot, 2).transpose((0, 2, 1)) 
data = data - np.mean(data, axis=-1, keepdims=True)   
labels = np.array([0] * slot_num + [1] * slot_num + [2] * slot_num + [3] * slot_num) 

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)
X_train_tensor, y_train_tensor = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
X_test_tensor, y_test_tensor = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# train with Conv1d and LSTM
class StiffModel(nn.Module):
    def __init__(self, input_channels, hidden_dim, output_size):
        super(StiffModel, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, stride=1) 
        self.bn1 = nn.BatchNorm1d(64)     
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1)  
        self.bn2 = nn.BatchNorm1d(128) 
        self.dropout = nn.Dropout(0.3)
        
        self.layer_norm = nn.LayerNorm(128)                
        self.lstm = nn.LSTM(128, hidden_dim, batch_first=True)              
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):  

        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        x =  torch.transpose(x, 1, 2)   
        x = self.layer_norm(x)
        lstm_out, (h_n, c_n) = self.lstm(x)  
        lstm_out = lstm_out[:, -1, :] 
        output = self.fc(lstm_out)
        
        return output

input_channels, hidden_dim, output_size = 2, 64, 4
model = StiffModel(input_channels, hidden_dim, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
train_loss, test_loss = [], []
loss_record, accuracy_record = [], []
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    
    for batch_data, batch_labels in train_loader:
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        train_loss.append(loss.detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Test set evaluation
    model.eval()

    test_loss = 0.0
    test_correct = 0
    test_total = 0
        
    with torch.no_grad():
        for test_data, test_labels in test_loader:
            test_outputs = model(test_data)
            test_loss += criterion(test_outputs, test_labels).item()
            _, test_predicted = torch.max(test_outputs, 1)
            test_total += test_labels.size(0)
            test_correct += (test_predicted == test_labels).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * test_correct / test_total

    print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
            
    loss_record.append(avg_test_loss)
    accuracy_record.append(test_accuracy)
     
np.savetxt('loss_record.txt', loss_record, delimiter=',', fmt='%1.4e')  
np.savetxt('accuracy_record.txt', accuracy_record, delimiter=',', fmt='%1.4e')  
       
# make the plot
epochs = list(range(1, num_epochs+1)) 

plt.figure(figsize=(6, 4))
plt.plot(epochs, loss_record, label="Test Loss", color='blue', linestyle='-', marker='o')
plt.plot(epochs, np.array(accuracy_record) / 100, label="Test Accuracy", color='red', linestyle='--', marker='x')

plt.xlim(-1,51)
plt.yticks([i * 0.1 for i in range(1,11)])

plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

plt.xlabel("Epochs")
plt.ylabel("Loss / Accuracy (Scaled)")
plt.legend()

plt.savefig("./performance.png", dpi=300)
plt.show()

