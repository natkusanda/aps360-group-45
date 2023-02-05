# %%
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdChemReactions
import numpy as np
import torch
from torch import nn, Tensor
import matplotlib.pyplot as plt 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import math

# %% 
class Net(nn.Module):
    # Constructor
    def __init__(self, Layers):
        super(Net, self).__init__()
        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(Layers, Layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))
    
    # Prediction
    def forward(self, activation):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                activation = F.relu(linear_transform(activation))    
            else:
                activation = linear_transform(activation)
        return activation
# %%
class NetRelu(nn.Module):
    
    # Constructor
    def __init__(self, D_in, H1, H2, H3, H4, D_out):
        super(NetRelu, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, H3)
        self.linear4 = nn.Linear(H3, H4)
        self.linear5 = nn.Linear(H4, D_out)
    
    # Prediction
    def forward(self, x):
        x = torch.relu(self.linear1(x))  
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = torch.relu(self.linear4(x))
        x = self.linear5(x)
        return x.double()
# %%
def get_mol_info(smi: str) -> np.array:
    mol = Chem.MolFromSmiles(smi)
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    rotors = Lipinski.NumRotatableBonds(mol)
    ap = len(mol.GetAromaticAtoms()) / mol.GetNumAtoms()

    return torch.tensor([mw, logp, rotors, ap])

# %%
def train(data_set, model, criterion, train_loader, validation_loader, optimizer, epochs=100):
    LOSS = []
    ACC = []
    for epoch in tqdm(range(epochs)):
        epoch_train_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            #LOSS.append(loss.item())
        LOSS.append(epoch_train_loss / len(train_loader))

        epoch_val_loss = 0
        for x, y in validation_loader:
            val_yhat = model(x)
            val_loss = criterion(val_yhat, y)
            epoch_val_loss += val_loss.item()
            #ACC.append(val_loss.item())
        ACC.append(epoch_val_loss / len(validation_loader))

    
    fig, ax2 = plt.subplots()
    color = 'tab:blue'
    ax2.set_ylabel('val loss', color = color) 
    ax2.set_xlabel('Iteration', color = color)
    ax2.plot(ACC, color = color)
    ax2.tick_params(axis = 'y', color = color)

    ax1 = ax2.twinx()  
    color = 'tab:red'
    ax1.plot(LOSS, color = color)
    ax1.set_ylabel('train loss', color = color)
    ax1.tick_params(axis = 'y', color = color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.savefig('/Users/nkusanda/Desktop/RESEARCH/morgan_nn/loss.jpg')
    plt.show()

    return LOSS, ACC

# %%
pwd = "aps360-group-45/"
df = pd.read_csv(pwd + "delaney.csv")
smiles = df['SMILES'].to_numpy()
sol = df['LogS'].to_numpy()

fps_delaney = []
for i in tqdm(range(len(smiles))):
    mol = Chem.MolFromSmiles(smiles[i])
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    rotors = Lipinski.NumRotatableBonds(mol)
    ap = len(mol.GetAromaticAtoms()) / mol.GetNumAtoms()

    x = torch.tensor([mw, logp, rotors, ap])
    y = torch.tensor((sol[i]))
    y = y.view(1)

    fps_delaney.append((x,y))

pwd = "aps360-group-45/"
df = pd.read_csv(pwd + "dls_100_unique.csv")
smiles_dls = df['SMILES'].to_numpy()
sol_dls = df['LogS'].to_numpy()

fps_dls100 = []
for i in tqdm(range(len(smiles_dls))):
    mol = Chem.MolFromSmiles(smiles_dls[i])
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    rotors = Lipinski.NumRotatableBonds(mol)
    ap = len(mol.GetAromaticAtoms()) / mol.GetNumAtoms()

    x = torch.tensor([mw, logp, rotors, ap])
    y = torch.tensor((sol_dls[i]))
    y = y.view(1)

    fps_dls100.append((x,y))

# %%
criterion = nn.MSELoss()
train_loader = torch.utils.data.DataLoader(dataset=fps_delaney, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=fps_dls100, batch_size=4, shuffle=False)
model = NetRelu(4, 20, 20, 20, 20, 1)
learning_rate = 0.00005
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
LOSS, ACC = train(data_set, model, criterion, train_loader, validation_loader, optimizer, epochs=10000)

# %%
predict = []
model.eval()
for i in tqdm(range(len(smiles_dls))):
    fp = get_mol_info(smiles_dls[i])
    pred = model(fp).detach().numpy()
    predict.append((smiles_dls[i], sol_dls[i], pred[0]))
df_bp = pd.DataFrame(predict, columns =['smile', 'logS', 'prediction'])
df_bp.to_csv(pwd + "predict_dls.csv")

# %%
