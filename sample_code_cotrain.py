import torch
import torch.nn as nn
import torch.optim as optim


# Define the SharedModel and ModelC
class SharedModel(nn.Module):
    def __init__(self):
        super(SharedModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 512)  # Output dimension is 512
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Define the downstream models with new names
class Model_WC_Main(nn.Module):
    def __init__(self):
        super(Model_WC_Main, self).__init__()
        self.fc1 = nn.Linear(512, 30)
        self.fc2 = nn.Linear(30, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class Model_WC_SparseSocial(nn.Module):
    def __init__(self):
        super(Model_WC_SparseSocial, self).__init__()
        self.fc1 = nn.Linear(512, 40)
        self.fc2 = nn.Linear(40, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class Model_NFC_Main(Model_WC_Main):
    def __init__(self):
        super(Model_NFC_Main, self).__init__()
        self.fc1 = nn.Linear(512, 30)
        self.fc2 = nn.Linear(30, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class Model_NFC_SparseSocial(Model_WC_SparseSocial):
    def __init__(self):
        super(Model_NFC_SparseSocial, self).__init__()
        self.fc1 = nn.Linear(512, 30)
        self.fc2 = nn.Linear(30, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class Model_VDD_Main(Model_WC_Main):
    def __init__(self):
        super(Model_VDD_Main, self).__init__()
        self.fc1 = nn.Linear(512, 30)
        self.fc2 = nn.Linear(30, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class Model_VDD_SparseSocial(Model_WC_SparseSocial):
    def __init__(self):
        super(Model_VDD_SparseSocial, self).__init__()
        self.fc1 = nn.Linear(512, 30)
        self.fc2 = nn.Linear(30, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Instantiate the models
shared_model = SharedModel()
model_wc_main = Model_WC_Main()
model_wc_sparse_social = Model_WC_SparseSocial()
model_nfc_main = Model_NFC_Main()
model_nfc_sparse_social = Model_NFC_SparseSocial()
model_vdd_main = Model_VDD_Main()
model_vdd_sparse_social = Model_VDD_SparseSocial()

# Define optimizers for all models
optimizer_shared = optim.SGD(shared_model.parameters(), lr=0.01)
optimizer_wc_main = optim.SGD(model_wc_main.parameters(), lr=0.01)
optimizer_wc_sparse_social = optim.SGD(model_wc_sparse_social.parameters(), lr=0.01)
optimizer_nfc_main = optim.SGD(model_nfc_main.parameters(), lr=0.01)
optimizer_nfc_sparse_social = optim.SGD(model_nfc_sparse_social.parameters(), lr=0.01)
optimizer_vdd_main = optim.SGD(model_vdd_main.parameters(), lr=0.01)
optimizer_vdd_sparse_social = optim.SGD(model_vdd_sparse_social.parameters(), lr=0.01)

# Input data could come from multiple tables
input_data_batch = pass

# Co-train the models on the dummy data
for x, target, src in range(len(input_data_batch)):
    # Forward pass through SharedModel for each input_data
    shared_output = shared_model(x)
    
    # Forward pass through downstream models using output of SharedModel as input
    if src == 'WC_Main':
        output = model_wc_main(x, shared_output)
        loss = criterion_wc_main(output, target)
        total_loss = total_loss + loss
    if src == 'WC_SparseSocial':
        output = model_wc_main(x, shared_output)
        loss = criterion_wc_sparse_social(output, target)
        total_loss = total_loss + loss
    # likewise others...
    
# Backward pass on the total loss
total_loss.backward()

# Optimize
optimizer_shared.step()
optimizer_wc_main.step()
optimizer_wc_sparse_social.step()